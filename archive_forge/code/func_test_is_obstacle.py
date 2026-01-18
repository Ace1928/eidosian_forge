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
def test_is_obstacle(self):
    node = Node(1, 1)
    self.assertFalse(self.a_star.is_obstacle(node))