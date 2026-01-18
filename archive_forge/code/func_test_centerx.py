import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_centerx(self):
    """Changing the centerx attribute moves the rect and does not change
        the rect's width
        """
    r = Rect(1, 2, 3, 4)
    new_centerx = r.centerx + 20
    expected_left = r.left + 20
    old_width = r.width
    r.centerx = new_centerx
    self.assertEqual(new_centerx, r.centerx)
    self.assertEqual(expected_left, r.left)
    self.assertEqual(old_width, r.width)