from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def get_tile_font_color(value: int) -> tuple:
    """
    Generates the font color for the tile based on its value.

    Args:
        value (int): The value of the tile.

    Returns:
        tuple: The font color (R, G, B) for the tile.
    """
    if value < 8:
        return (119, 110, 101)
    return (249, 246, 242)