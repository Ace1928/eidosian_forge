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
def shift_and_combine(row: list) -> Tuple[list, int]:
    """
            Shifts non-zero elements to the left and combines elements of the same value.
            Args:
                row (list): A row (or column) from the game board.
            Returns:
                Tuple[list, int]: The shifted and combined row, and the score gained.
            """
    non_zero = [i for i in row if i != 0]
    combined = []
    score = 0
    skip = False
    for i in range(len(non_zero)):
        if skip:
            skip = False
            continue
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
            combined.append(2 * non_zero[i])
            score += 2 * non_zero[i]
            skip = True
        else:
            combined.append(non_zero[i])
    combined.extend([0] * (len(row) - len(combined)))
    return (combined, score)