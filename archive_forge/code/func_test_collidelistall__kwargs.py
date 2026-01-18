import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidelistall__kwargs(self):
    r = Rect(1, 1, 10, 10)
    l = [Rect(1, 1, 10, 10), Rect(5, 5, 10, 10), Rect(15, 15, 1, 1), Rect(2, 2, 1, 1)]
    self.assertEqual(r.collidelistall(l), [0, 1, 3])
    f = [Rect(50, 50, 1, 1), Rect(20, 20, 5, 5)]
    self.assertFalse(r.collidelistall(rects=f))