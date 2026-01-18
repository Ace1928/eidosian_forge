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
def calculate_path_cost(self, path):
    """
        Calculates the cost of a given path based on length, proximity to dangers, and strategic advantages.
        This method employs a detailed and comprehensive cost analysis to ensure optimal path selection.

        Args:
            path (list): The path as a list of grid positions.

        Returns:
            float: The calculated cost of the path.
        """
    length_cost = len(path) * 10
    danger_cost = sum((self.grid.is_near_obstacle(point) for point in path)) * 20
    strategic_cost = self.evaluate_strategic_advantages(path)
    return length_cost + danger_cost + strategic_cost