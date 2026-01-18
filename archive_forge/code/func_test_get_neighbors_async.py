import unittest
import math
from A_STAR import A_STAR
from pygame.math import Vector2
from typing import List, Tuple, Any, Set
from Utility import Grid, Node
from Snake import Snake
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
def test_get_neighbors_async(self):
    node = Node(1, 1)
    neighbors = asyncio.run(self.a_star.get_neighbors_async(node))
    self.assertIsInstance(neighbors, list)
    self.assertEqual(len(neighbors), 4)