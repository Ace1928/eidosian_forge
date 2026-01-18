import numpy as np
from ai_logic import (
from gui_utils import (
from typing import List, Tuple
import types
import importlib.util
import logging
import random
@StandardDecorator()
def on_game_move(board: np.ndarray, move: str):
    """
    Performs tasks when a move is made in the game.

    Args:
        board (np.ndarray): The current game board state.
        move (str): The move made ('up', 'down', 'left', 'right').
    """