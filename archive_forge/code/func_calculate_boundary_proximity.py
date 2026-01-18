from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint
def calculate_boundary_proximity(self, position: Tuple[int, int], boundaries: Tuple[int, int, int, int], space_around_boundaries: int) -> float:
    """
        Calculate a penalty based on the proximity to boundaries.

        Args:
            position: The current position as a tuple of (x, y) coordinates.
            boundaries: The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max).
            space_around_boundaries: The desired space to maintain around boundaries.

        Returns:
            The calculated penalty based on proximity to boundaries.
        """
    x_min, y_min, x_max, y_max = boundaries
    min_dist_to_boundary = min(position[0] - x_min, x_max - position[0], position[1] - y_min, y_max - position[1])
    if min_dist_to_boundary < space_around_boundaries:
        return (space_around_boundaries - min_dist_to_boundary) ** 2
    return 0.0