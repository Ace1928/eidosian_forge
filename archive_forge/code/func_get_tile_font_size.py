from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def get_tile_font_size(value: int) -> int:
    """
    Generates the font size for the tile based on its value.

    Args:
        value (int): The value of the tile.

    Returns:
        int: The font size for the tile.
    """
    if value < 100:
        return 55
    if value < 1000:
        return 45
    if value < 10000:
        return 35
    return 25