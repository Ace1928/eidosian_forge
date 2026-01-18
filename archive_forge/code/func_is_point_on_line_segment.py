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
def is_point_on_line_segment(self, point, p1, p2):
    """
        Determines whether a given point lies on the line segment between two other points.
        This method employs a precise geometric calculation to check if the point coincides with the line segment,
        considering floating-point precision and ensuring reliable results.

        Args:
            point (tuple): The point to check.
            p1 (tuple): The first endpoint of the line segment.
            p2 (tuple): The second endpoint of the line segment.

        Returns:
            bool: True if the point lies on the line segment, False otherwise.
        """
    cross_product = (point[1] - p1[1]) * (p2[0] - p1[0]) - (point[0] - p1[0]) * (p2[1] - p1[1])
    if abs(cross_product) > 1e-10:
        return False
    dot_product = (point[0] - p1[0]) * (p2[0] - p1[0]) + (point[1] - p1[1]) * (p2[1] - p1[1])
    if dot_product < 0:
        return False
    squared_length = (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1])
    if dot_product > squared_length:
        return False
    return True