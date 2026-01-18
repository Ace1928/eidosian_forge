from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def draw_ui() -> None:
    """
    Draws the user interface for the game.

    This function is responsible for rendering the GUI elements on the screen.
    """