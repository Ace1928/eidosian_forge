import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidelistall_returns_empty_list(self):
    r = Rect(1, 1, 10, 10)
    l = [Rect(112, 1, 10, 10), Rect(50, 5, 10, 10), Rect(15, 15, 1, 1), Rect(-20, 2, 1, 1)]
    self.assertEqual(r.collidelistall(l), [])