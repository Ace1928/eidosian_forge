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
def test_calculate_heuristic(self):
    goalstate = Node(4, 4)
    neighbor = Node(1, 1)
    heuristic = self.a_star.calculate_heuristic(goalstate, neighbor)
    self.assertEqual(heuristic, 3.5, 0.4)