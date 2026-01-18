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
def apply_glow(self, WIN, position, glow_color):
    """
        Applies a glow effect around a given position with the specified glow color.
        """
    glow_radius = 10
    for radius in range(glow_radius, 0, -1):
        alpha = (1 - radius / glow_radius) * 255
        glow_surface = pygame.Surface((TILE_SIZE + radius * 2, TILE_SIZE + radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, glow_color + (int(alpha),), (glow_radius, glow_radius), radius)
        WIN.blit(glow_surface, (position[0] * TILE_SIZE - radius, position[1] * TILE_SIZE - radius))
    pygame.display.update()