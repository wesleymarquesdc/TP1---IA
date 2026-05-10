from typing import List, Tuple, Optional, Dict, Tuple
import time
import math
import random

ROWS, COLS = 6, 7
EMPTY, P1, P2 = 0, 1, 2

# -----------------------------------------------------------------------------
# Utilidades de tabuleiro (PRONTAS)
# -----------------------------------------------------------------------------
def copy_board(board: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in board]

def valid_moves(board: List[List[int]]) -> List[int]:
    """Retorna as colunas ainda jogáveis (topo vazio)."""
    return [c for c in range(COLS) if board[0][c] == EMPTY]

def make_move(board: List[List[int]], col: int, player: int) -> Optional[List[List[int]]]:
    """Retorna um novo tabuleiro aplicando a gravidade na coluna col; None se inválido."""
    if col < 0 or col >= COLS or board[0][col] != EMPTY:
        return None
    nb = copy_board(board)
    for r in reversed(range(ROWS)):
        if nb[r][col] == EMPTY:
            nb[r][col] = player
            return nb
    return None

def winner(board: List[List[int]]) -> int:
    """0 se ninguém venceu; 1 ou 2 se há 4 em linha."""
    # Horizontais
    for r in range(ROWS):
        for c in range(COLS - 3):
            x = board[r][c]
            if x != EMPTY and x == board[r][c+1] == board[r][c+2] == board[r][c+3]:
                return x
    # Verticais
    for c in range(COLS):
        for r in range(ROWS - 3):
            x = board[r][c]
            if x != EMPTY and x == board[r+1][c] == board[r+2][c] == board[r+3][c]:
                return x
    # Diag ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            x = board[r][c]
            if x != EMPTY and x == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3]:
                return x
    # Diag ↗
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            x = board[r][c]
            if x != EMPTY and x == board[r-1][c+1] == board[r-2][c+2] == board[r-3][c+3]:
                return x
    return 0

def is_full(board: List[List[int]]) -> bool:
    return all(board[0][c] != EMPTY for c in range(COLS))

def terminal(board: List[List[int]]) -> Tuple[bool, int]:
    """(é_terminal, vencedor) com vencedor=0 para empate/indefinido."""
    w = winner(board)
    if w != 0:
        return True, w
    if is_full(board):
        return True, 0
    return False, 0

def other(player: int) -> int:
    return P1 if player == P2 else P2

# -----------------------------------------------------------------------------
# Algoritmo MINIMAX
# -----------------------------------------------------------------------------
def evaluate_window(window: List[int], max_player: int) -> float:
    min_player = other(max_player)

    max_player_count = window.count(max_player)
    min_player_count = window.count(min_player)
    empty_count = window.count(EMPTY)

    score = 0.0

    # max possui uma ameaça imediata
    if max_player_count == 3 and empty_count == 1:
        score += 5

    # max possui uma dupla (ameaça futura)
    elif max_player_count == 2 and empty_count == 2:
        score += 0.2

    # min possui uma ameaça imediata
    if min_player_count == 3 and empty_count == 1:
        score -= 10

    # min possui uma dupla (ameaça futura)
    elif min_player_count == 2 and empty_count == 2:
        score -= 0.3

    return score

def heuristic(board: List[List[int]], max_player: int) -> float:
    score = 0.0

    # Horizontais
    for r in range(ROWS):
        for c in range(COLS - 3):

            window = [
                board[r][c],
                board[r][c + 1],
                board[r][c + 2],
                board[r][c + 3]
            ]

            score += evaluate_window(window, max_player)

    # Verticais
    for c in range(COLS):
        for r in range(ROWS - 3):

            window = [
                board[r][c],
                board[r + 1][c],
                board[r + 2][c],
                board[r + 3][c]
            ]

            score += evaluate_window(window, max_player)

    # Dia ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):

            window = [
                board[r][c],
                board[r + 1][c + 1],
                board[r + 2][c + 2],
                board[r + 3][c + 3]
            ]

            score += evaluate_window(window, max_player)

    # Diag ↗
    for r in range(3, ROWS):
        for c in range(COLS - 3):

            window = [
                board[r][c],
                board[r - 1][c + 1],
                board[r - 2][c + 2],
                board[r - 3][c + 3]
            ]

            score += evaluate_window(window, max_player)

    return math.tanh(score)

def get_terminal_value(winner: int, max_player: int) -> int:
    if winner:
        return 1 if winner == max_player else -1
    return 0

def minimax_decision(board: List[List[int]], max_player: int, max_depth: int):
    best_value = float('-inf')
    best_move = None

    depth = 0
    for move in valid_moves(board):
        new_board = make_move(board, move, max_player)

        value = min_value(new_board, max_player, depth + 1, max_depth)

        if value > best_value:
            best_value = value
            best_move = move

    return best_move

def max_value(board: List[List[int]], max_player: int, depth: int, max_depth: int) -> int:
    is_terminal, winner = terminal(board)
    if is_terminal:
        return get_terminal_value(winner, max_player)
    
    if depth >= max_depth:
        return heuristic(board, max_player)
    
    value = float('-inf')
    for max_move in valid_moves(board):
        new_board = make_move(board, max_move, max_player)

        value = max(
            value,
            min_value(new_board, max_player, depth + 1, max_depth)
        )

    return value

def min_value(board: List[List[int]], max_player: int, depth: int, max_depth: int) -> int:
    is_terminal, winner = terminal(board)
    if is_terminal:
        return get_terminal_value(winner, max_player)
    
    if depth >= max_depth:
        return heuristic(board, max_player)

    min_player = other(max_player)

    value = float('inf')
    for min_move in valid_moves(board):
        new_board = make_move(board, min_move, min_player)

        value = min(
            value,
            max_value(new_board, max_player, depth + 1, max_depth)
        )

    return value

# -----------------------------------------------------------------------------
# ÚNICO PONTO A SER IMPLEMENTADO PELOS ALUNOS
# -----------------------------------------------------------------------------
def choose_move(board: List[List[int]], turn: int, config: Dict) -> Tuple[int, Dict]:
    """
    Decide a coluna (0..6) para jogar agora.

    Parâmetros:
      - board: matriz 6x7 com valores {0,1,2}
      - turn: 1 ou 2
      - config: {"max_time_ms": int, "max_depth": int}

    Retorna:
      - col: int (0..6)
    """
    max_time_ms = int(config.get("max_time_ms"))
    max_depth = int(config.get("max_depth"))
    turn = int(turn)

    print(f"AI choose_move called with max_time_ms={max_time_ms}, max_depth={max_depth}, player={turn}")
    
    start = time.time()

    # Função auxiliar para checar tempo decorrido   
    def time_exceeded():
        return max_time_ms > 0 and (time.time() - start) * 1000.0 >= max_time_ms
    
    legal = valid_moves(board)

    move = 0
    if not legal:
        # Sem jogadas: devolve 0 por convenção (servidor lida com isso)
        return move

    move = minimax_decision(board, turn, max_depth)

    return move

def choose_move_randomly(board: List[List[int]], turn: int, config: Dict) -> Tuple[int, Dict]:
    max_time_ms = int(config.get("max_time_ms"))
    max_depth = int(config.get("max_depth"))
    turn = int(turn)

    print(f"AI choose_move called with max_time_ms={max_time_ms}, max_depth={max_depth}, player={turn}")
    
    legal = valid_moves(board)

    move = 0
    if not legal:
        return move
    
    move = random.choice(legal)
    return move


def choose_move_infinity(board: List[List[int]], turn: int, config: Dict) -> Tuple[int, Dict]:
    """
    Decide a coluna (0..6) para jogar agora.

    Parâmetros:
      - board: matriz 6x7 com valores {0,1,2}
      - turn: 1 ou 2
      - config: {"max_time_ms": int, "max_depth": int}

    Retorna:
      - col: int (0..6)
    """
    max_time_ms = int(config.get("max_time_ms"))
    max_depth = int(config.get("max_depth"))
    turn = int(turn)

    print(f"AI choose_move called with max_time_ms={max_time_ms}, max_depth={max_depth}, player={turn}")
    
    start = time.time()

    # Função auxiliar para checar tempo decorrido   
    def time_exceeded():
        return max_time_ms > 0 and (time.time() - start) * 1000.0 >= max_time_ms
    
    legal = valid_moves(board)

    move = 0
    if not legal:
        # Sem jogadas: devolve 0 por convenção (servidor lida com isso)
        return move
    
    # VERSÃO INICIAL: escolhe aleatoriamente entre as jogadas legais
    i = 0
    while True:
        i += 1

    return move