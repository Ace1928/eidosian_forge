import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_left(self):
    """Changing the left attribute moves the rect and does not change
        the rect's width
        """
    r = Rect(1, 2, 3, 4)
    new_left = 10
    r.left = new_left
    self.assertEqual(new_left, r.left)
    self.assertEqual(Rect(new_left, 2, 3, 4), r)