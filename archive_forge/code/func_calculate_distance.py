from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint
def calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
        Calculate the Manhattan distance between two points.

        Args:
            pos1: The first position as a tuple of (x, y) coordinates.
            pos2: The second position as a tuple of (x, y) coordinates.

        Returns:
            The Manhattan distance between the two points.
        """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])