import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline__no_overlap(self):
    """Ensures lines that do not overlap the rect are not clipped."""
    rect = Rect((10, 25), (15, 20))
    big_rect = rect.inflate(2, 2)
    lines = ((big_rect.bottomleft, big_rect.topleft), (big_rect.topleft, big_rect.topright), (big_rect.topright, big_rect.bottomright), (big_rect.bottomright, big_rect.bottomleft))
    expected_line = ()
    for line in lines:
        clipped_line = rect.clipline(line)
        self.assertTupleEqual(clipped_line, expected_line)