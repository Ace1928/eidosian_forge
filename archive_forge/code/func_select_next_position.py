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
def select_next_position(self, current_position, visited):
    """
        Selects the next position for the amoeba-inspired Hamiltonian pathfinding algorithm.
        This method carefully evaluates the candidate positions surrounding the current position, considering
        factors such as grid boundaries, obstacles, the snake's body, and previously visited cells, to determine
        the most promising next step in the path.

        Args:
            current_position (tuple): The current position of the pathfinding algorithm.
            visited (set): A set of previously visited positions.

        Returns:
            tuple: The selected next position for the pathfinding algorithm, chosen based on a comprehensive
            assessment of the available options and their potential to lead to a complete Hamiltonian path.
        """
    candidate_positions = self.get_candidate_positions(current_position)
    valid_positions = [pos for pos in candidate_positions if self.is_valid_position(pos) and pos not in visited]
    if not valid_positions:
        return None
    next_position = random.choice(valid_positions)
    logging.debug(f'Selected next position: {next_position}')
    return next_position