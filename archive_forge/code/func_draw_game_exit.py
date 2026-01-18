from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def draw_game_exit() -> None:
    """
    Draws the game exit screen to the GUI for visualization.
    """