import numpy as np  # Assuming NumPy is used for efficient array manipulation
import random
import types
import importlib.util
import logging
import collections
from typing import Deque, Dict, Tuple, List
from typing import List, Tuple
from functools import wraps
import logging
@StandardDecorator()
def dynamic_depth_expectimax(board: np.ndarray, playerTurn: bool, initial_depth: int=3) -> Tuple[float, str]:
    """
    Calculates the best move using the expectimax algorithm with dynamic depth adjustment based on the game state complexity.

    Args:
        board (np.ndarray): The current game board.
        playerTurn (bool): Flag indicating whether it's the player's turn or a chance node.
        initial_depth (int): The initial depth for the search, adjusted dynamically.

    Returns:
        Tuple[float, str]: The best heuristic value found and the corresponding move.
    """
    logging.info(f'Starting dynamic depth expectimax with initial depth {initial_depth}.')
    empty_tiles = len(get_empty_tiles(board))
    depth = adjust_depth_based_on_complexity(initial_depth, empty_tiles)
    logging.info(f'Adjusted depth based on complexity: {depth}.')
    best_heuristic_value, best_move = expectimax(board, depth, playerTurn)
    logging.info(f'Best heuristic value: {best_heuristic_value}, Best move: {best_move}.')
    return (best_heuristic_value, best_move)