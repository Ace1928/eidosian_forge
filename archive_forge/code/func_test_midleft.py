import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_midleft(self):
    """Changing the midleft attribute moves the rect and does not change
        the rect's size
        """
    r = Rect(1, 2, 3, 4)
    new_midleft = (r.left + 20, r.centery + 30)
    expected_topleft = (r.left + 20, r.top + 30)
    old_size = r.size
    r.midleft = new_midleft
    self.assertEqual(new_midleft, r.midleft)
    self.assertEqual(expected_topleft, r.topleft)
    self.assertEqual(old_size, r.size)