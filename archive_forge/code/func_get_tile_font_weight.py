from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def get_tile_font_weight(value: int) -> str:
    """
    Generates the font weight for the tile based on its value.

    Args:
        value (int): The value of the tile.

    Returns:
        str: The font weight for the tile.
    """
    if value < 100:
        return 'bold'
    return 'normal'