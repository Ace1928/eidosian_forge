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
def is_position_free(self, position):
    """
        Checks if a given position is free (not occupied) on the grid.

        Args:
            position (numpy array): The position to check on the grid.

        Returns:
            bool: True if the position is free, False otherwise.
        """
    x, y = position
    return self.cells[y, x] == 0