import numpy as np
from ai_logic import (
from gui_utils import (
from typing import List, Tuple
import types
import importlib.util
import logging
import random
@StandardDecorator()
def randomise_next_tile(board: np.ndarray) -> np.ndarray:
    """
    Randomly selects a position on the board and places a new tile (2 or 4) at that position.

    Args:
        board (np.ndarray): The current game board.

    Returns:
        np.ndarray: The updated game board with the new tile added.
    """
    empty_positions = list(zip(*np.where(board == 0)))
    if empty_positions:
        x, y = random.choice(empty_positions)
        board[x, y] = 2 if random.random() < 0.9 else 4
    return board