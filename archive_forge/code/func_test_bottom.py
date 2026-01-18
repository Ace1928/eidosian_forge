import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_bottom(self):
    """Changing the bottom attribute moves the rect and does not change
        the rect's height
        """
    r = Rect(1, 2, 3, 4)
    new_bottom = r.bottom + 20
    expected_top = r.top + 20
    old_height = r.height
    r.bottom = new_bottom
    self.assertEqual(new_bottom, r.bottom)
    self.assertEqual(expected_top, r.top)
    self.assertEqual(old_height, r.height)