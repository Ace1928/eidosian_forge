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
def generate_random_position(self):
    """
        Generates a random position for the food within the grid boundaries.

        Returns:
            numpy array: The generated random position.
        """
    x = np.random.randint(0, WIDTH)
    y = np.random.randint(0, HEIGHT)
    position = np.array([x, y])
    logging.debug(f'Generated random food position: {position}')
    return position