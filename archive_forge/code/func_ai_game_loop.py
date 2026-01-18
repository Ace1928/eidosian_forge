from ai_logic import (
from gui_utils import (
from game_manager import (
from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
@StandardDecorator()
def ai_game_loop(board: np.ndarray, depth: int=3) -> Tuple[np.ndarray, int]:
    """
    Runs the game loop using an AI agent to make moves.

    Args:
        board (np.ndarray): The current game board.
        depth (int): The depth of the search tree for the AI agent.

    Returns:
        Tuple[np.ndarray, int]: The final game board state and the total score.
    """
    score = 0
    game_over = False
    while not game_over:
        best_move = calculate_best_move(board)
        board, move_score = simulate_move(board, best_move)
        score += move_score
        game_over = is_game_over(board)
        if not game_over:
            add_random_tile(board)
    return (board, score)