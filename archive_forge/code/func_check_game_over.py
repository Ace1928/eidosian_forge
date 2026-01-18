import numpy as np
from ai_logic import (
from gui_utils import (
from typing import List, Tuple
import types
import importlib.util
import logging
import random
@StandardDecorator()
def check_game_over(board: np.ndarray) -> bool:
    """
    Checks if the game is over by determining if there are any valid moves left.

    Args:
        board (np.ndarray): The current game board.

    Returns:
        bool: True if the game is over, False otherwise.
    """
    for move in ['up', 'down', 'left', 'right']:
        if simulate_move(board, move)[0].tolist() != board.tolist():
            return False
    return True