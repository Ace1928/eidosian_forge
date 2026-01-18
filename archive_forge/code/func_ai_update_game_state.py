from ai_logic import (
from gui_utils import (
from game_manager import (
from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
@StandardDecorator()
def ai_update_game_state(board: np.ndarray, depth: int=3) -> Tuple[np.ndarray, int]:
    """
    Updates the game state using an AI agent to make moves.

    Args:
        board (np.ndarray): The current game board.
        depth (int): The depth of the search tree for the AI agent.

    Returns:
        Tuple[np.ndarray, int]: The updated game board and the total score.
    """
    return ai_game_loop(board, depth)