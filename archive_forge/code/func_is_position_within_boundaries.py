from typing import (
import pygame  # Import pygame module for game development
from random import (
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality
import math  # Import math module for mathematical operations
def is_position_within_boundaries(self, position: Tuple[int, int], boundaries: Tuple[int, int, int, int]=(0, 0, 100, 100)) -> bool:
    """
        Check if a position is within the specified boundaries.

        This function determines whether a given position falls within the defined environmental boundaries.

        Args:
            position (Tuple[int, int]): The position to check as a tuple.
            boundaries (Tuple[int, int, int, int]): The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max). Default is (0, 0, 100, 100).

        Returns:
            bool: True if the position is within the boundaries, False otherwise.
        """
    x_min, y_min, x_max, y_max = boundaries
    return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max