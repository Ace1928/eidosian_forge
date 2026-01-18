import pygame as pg
import sys
from random import randint, seed
from collections import deque
from typing import List, Tuple, Deque, Optional, Set, Dict
import logging
import math
from queue import PriorityQueue
def calculate_next_position(self) -> Tuple[int, int]:
    """
        Calculates the next position of the snake using the Theta* pathfinding algorithm.
        """
    path = self.grid.pathfinding.theta_star_path(self.body[0], self.fruit.position, self.body, self.grid)
    return path[0] if path else self.body[0]