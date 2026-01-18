import numpy as np
from ai_logic import (
from gui_utils import (
from typing import List, Tuple
import types
import importlib.util
import logging
import random
@StandardDecorator()
def initialize_game() -> np.ndarray:
    """
    Initializes the game board.

    Returns:
        np.ndarray: The initialized game board as a 2D NumPy array.
    """
    board = np.zeros((4, 4), dtype=int)
    add_random_tile(board)
    add_random_tile(board)
    return board