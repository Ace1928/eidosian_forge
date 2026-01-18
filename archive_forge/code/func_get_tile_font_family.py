from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def get_tile_font_family(value: int) -> str:
    """
    Generates the font family for the tile based on its value.

    Args:
        value (int): The value of the tile.

    Returns:
        str: The font family for the tile.
    """
    return 'Verdana' if value < 1000 else 'Arial'