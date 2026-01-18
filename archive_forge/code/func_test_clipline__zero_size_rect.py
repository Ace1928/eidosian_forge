import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline__zero_size_rect(self):
    """Ensures clipline handles zero sized rects correctly."""
    expected_line = ()
    for size in ((0, 15), (15, 0), (0, 0)):
        rect = Rect((10, 25), size)
        clipped_line = rect.clipline(rect.topleft, rect.topleft)
        self.assertTupleEqual(clipped_line, expected_line)