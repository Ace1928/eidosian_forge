import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline(self):
    """Ensures clipline handles four int parameters.

        Tests the clipline(x1, y1, x2, y2) format.
        """
    rect = Rect((1, 2), (35, 40))
    x1 = 5
    y1 = 6
    x2 = 11
    y2 = 19
    expected_line = ((x1, y1), (x2, y2))
    clipped_line = rect.clipline(x1, y1, x2, y2)
    self.assertIsInstance(clipped_line, tuple)
    self.assertTupleEqual(clipped_line, expected_line)