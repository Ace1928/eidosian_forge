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
def evaluate_obstacle_proximity_cost(self, position):
    """
        Evaluates the cost associated with the proximity of a given position to obstacles on the grid.
        This method quantifies the risk and potential impact of being close to obstacles during gameplay.

        Args:
            position (tuple): The position to evaluate.

        Returns:
            float: The obstacle proximity cost for the given position.
        """
    obstacle_positions = self.grid.get_obstacles()
    min_distance = min((np.linalg.norm(np.array(position) - np.array(obstacle)) for obstacle in obstacle_positions))
    cost = 20 / (min_distance + 1)
    logging.debug(f'Calculated obstacle proximity cost for position {position}: {cost}')
    return cost