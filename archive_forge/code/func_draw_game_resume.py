from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def draw_game_resume() -> None:
    """
    Draws the game resume screen to the GUI for visualization.
    """