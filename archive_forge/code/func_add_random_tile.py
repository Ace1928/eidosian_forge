import numpy as np
from ai_logic import (
from gui_utils import (
from typing import List, Tuple
import types
import importlib.util
import logging
import random
@StandardDecorator()
def add_random_tile(board: np.ndarray) -> None:
    """
    Adds a random tile (2 or 4) to an empty position on the board.

    Args:
        board (np.ndarray): The game board.
    """
    empty_positions = list(zip(*np.where(board == 0)))
    if empty_positions:
        x, y = random.choice(empty_positions)
        board[x, y] = 2 if random.random() < 0.9 else 4