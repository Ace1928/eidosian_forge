from ai_logic import (
from gui_utils import (
from game_manager import (
from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
@StandardDecorator()
def ai_game_over_check(board: np.ndarray, depth: int=3) -> bool:
    """
    Checks if the game is over for an AI agent.

    Args:
        board (np.ndarray): The current game board.
        depth (int): The depth of the search tree for the AI agent.

    Returns:
        bool: True if the game is over, False otherwise.
    """
    return is_game_over(board)