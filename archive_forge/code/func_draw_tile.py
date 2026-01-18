from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def draw_tile(tile_value: int) -> None:
    """
    Draws a tile with a specific value to the GUI for visualization.

    Args:
        tile_value (int): The value of the tile to be drawn.
    """