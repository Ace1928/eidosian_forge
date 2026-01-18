import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline__equal_endpoints_no_overlap(self):
    """Ensures clipline handles lines with both endpoints the same.

        Testing lines that do not overlap the rect.
        """
    expected_line = ()
    rect = Rect((10, 25), (15, 20))
    for pt in test_utils.rect_perimeter_pts(rect.inflate(2, 2)):
        clipped_line = rect.clipline((pt, pt))
        self.assertTupleEqual(clipped_line, expected_line)