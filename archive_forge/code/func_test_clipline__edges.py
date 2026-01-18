import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline__edges(self):
    """Ensures clipline properly clips line that are along the rect edges."""
    rect = Rect((10, 25), (15, 20))
    edge_dict = {(rect.bottomleft, rect.topleft): ((rect.bottomleft[0], rect.bottomleft[1] - 1), rect.topleft), (rect.topleft, rect.topright): (rect.topleft, (rect.topright[0] - 1, rect.topright[1])), (rect.topright, rect.bottomright): (), (rect.bottomright, rect.bottomleft): ()}
    for edge, expected_line in edge_dict.items():
        clipped_line = rect.clipline(edge)
        self.assertTupleEqual(clipped_line, expected_line)
        if expected_line:
            expected_line = (expected_line[1], expected_line[0])
        clipped_line = rect.clipline((edge[1], edge[0]))
        self.assertTupleEqual(clipped_line, expected_line)