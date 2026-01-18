import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidelist(self):
    r = Rect(1, 1, 10, 10)
    l = [Rect(50, 50, 1, 1), Rect(5, 5, 10, 10), Rect(15, 15, 1, 1)]
    self.assertEqual(r.collidelist(l), 1)
    f = [Rect(50, 50, 1, 1), (100, 100, 4, 4)]
    self.assertEqual(r.collidelist(f), -1)