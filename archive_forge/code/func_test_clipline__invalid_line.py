import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline__invalid_line(self):
    """Ensures clipline handles invalid lines correctly."""
    rect = Rect((0, 0), (10, 20))
    invalid_lines = ((), (1,), (1, 2), (1, 2, 3), (1, 2, 3, 4, 5), ((1, 2),), ((1, 2), (3,)), ((1, 2), 3), ((1, 2, 5), (3, 4)), ((1, 2), (3, 4, 5)), ((1, 2), (3, 4), (5, 6)))
    for line in invalid_lines:
        with self.assertRaises(TypeError):
            clipped_line = rect.clipline(line)
        with self.assertRaises(TypeError):
            clipped_line = rect.clipline(*line)