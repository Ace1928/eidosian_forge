from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def draw_game_help() -> None:
    """
    Draws the game help screen to the GUI for visualization.
    """