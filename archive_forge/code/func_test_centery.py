import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_centery(self):
    """Changing the centery attribute moves the rect and does not change
        the rect's width
        """
    r = Rect(1, 2, 3, 4)
    new_centery = r.centery + 20
    expected_top = r.top + 20
    old_height = r.height
    r.centery = new_centery
    self.assertEqual(new_centery, r.centery)
    self.assertEqual(expected_top, r.top)
    self.assertEqual(old_height, r.height)