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
def get_obstacles(self):
    """
        Returns a list of all obstacle positions on the grid.

        Returns:
            list of tuples: A list containing (x, y) coordinates of all obstacles on the grid.
        """
    obstacles = [(x, y) for x in range(self.width) for y in range(self.height) if self.cells[y, x] != 0]
    logging.debug(f'Found {len(obstacles)} obstacles on the grid')
    return obstacles