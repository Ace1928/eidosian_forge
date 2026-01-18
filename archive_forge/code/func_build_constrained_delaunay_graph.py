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
def build_constrained_delaunay_graph(self):
    """
        Builds a constrained Delaunay triangulation graph based on the given points and obstacles.
        This method employs advanced computational geometry techniques to construct a highly optimized and
        topologically sound graph structure, facilitating efficient pathfinding operations while strictly
        respecting the specified constraints.

        Returns:
            networkx.Graph: The constructed constrained Delaunay triangulation graph, meticulously crafted to
            provide a solid foundation for precise and reliable pathfinding algorithms.
        """
    logging.debug('Building constrained Delaunay triangulation graph.')
    try:
        tri = Delaunay(self.points)
        graph = nx.Graph()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    if not self.is_constrained_edge(tri.points[simplex[i]], tri.points[simplex[j]]):
                        graph.add_edge(tuple(tri.points[simplex[i]]), tuple(tri.points[simplex[j]]))
        logging.info(f'Constrained Delaunay triangulation graph constructed with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.')
        return graph
    except Exception as e:
        logging.error(f'Failed to build constrained Delaunay triangulation graph: {e}')
        raise Exception(f'Constrained Delaunay triangulation error: {e}')