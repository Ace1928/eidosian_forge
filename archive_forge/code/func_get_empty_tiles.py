import numpy as np  # Assuming NumPy is used for efficient array manipulation
import random
import types
import importlib.util
import logging
import collections
from typing import Deque, Dict, Tuple, List
from typing import List, Tuple
from functools import wraps
import logging
@StandardDecorator()
def get_empty_tiles(board: np.ndarray) -> List[Tuple[int, int]]:
    """
    Finds all empty tiles on the board.

    Args:
        board (np.ndarray): The game board.

    Returns:
        List[Tuple[int, int]]: A list of coordinates for the empty tiles.
    """
    logging.debug('Identifying empty tiles on the board.')
    empty_tiles = list(zip(*np.where(board == 0)))
    logging.debug(f'Found {len(empty_tiles)} empty tiles.')
    return empty_tiles