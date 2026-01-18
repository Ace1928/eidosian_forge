import pygame
import random
import heapq
import logging
from typing import List, Optional, Dict, Any, Tuple
import cProfile
from collections import deque
import numpy as np
import time
import torch
from functools import lru_cache as LRUCache
import math
import asyncio
from scipy.spatial import Delaunay
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from queue import PriorityQueue
from collections import defaultdict
def line_of_sight(self, start, end):
    """
        Determines whether there is a clear line of sight between two positions on the grid.
        This method uses the Bresenham's line algorithm to efficiently check for any obstacles or
        obstructions along the path, enabling the Theta* algorithm to make informed decisions about
        node expansion and path optimization.

        Args:
            start (tuple): The starting position of the line of sight check.
            end (tuple): The endpoint of the line of sight check.

        Returns:
            bool: True if there is a clear line of sight between the start and end positions, False otherwise.
        """
    x1, y1 = start
    x2, y2 = end
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while x1 != x2 or y1 != y2:
        if self.grid.is_obstacle((x1, y1)):
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return True