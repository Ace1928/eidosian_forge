import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_midtop(self):
    """Changing the midtop attribute moves the rect and does not change
        the rect's size
        """
    r = Rect(1, 2, 3, 4)
    new_midtop = (r.centerx + 20, r.top + 30)
    expected_topleft = (r.left + 20, r.top + 30)
    old_size = r.size
    r.midtop = new_midtop
    self.assertEqual(new_midtop, r.midtop)
    self.assertEqual(expected_topleft, r.topleft)
    self.assertEqual(old_size, r.size)