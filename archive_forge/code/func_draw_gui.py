from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def draw_gui(board: np.ndarray) -> None:
    """
    Draws the game board to the GUI for visualization.

    Args:
        board (np.ndarray): The current game board.
    """