import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline__negative_size_rect(self):
    """Ensures clipline handles negative sized rects correctly."""
    expected_line = ()
    for size in ((-15, 20), (15, -20), (-15, -20)):
        rect = Rect((10, 25), size)
        norm_rect = rect.copy()
        norm_rect.normalize()
        big_rect = norm_rect.inflate(2, 2)
        line_dict = {(big_rect.midleft, big_rect.midright): (norm_rect.midleft, (norm_rect.midright[0] - 1, norm_rect.midright[1])), (big_rect.midtop, big_rect.midbottom): (norm_rect.midtop, (norm_rect.midbottom[0], norm_rect.midbottom[1] - 1)), (big_rect.midleft, norm_rect.center): (norm_rect.midleft, norm_rect.center), (big_rect.midtop, norm_rect.center): (norm_rect.midtop, norm_rect.center), (big_rect.midright, norm_rect.center): ((norm_rect.midright[0] - 1, norm_rect.midright[1]), norm_rect.center), (big_rect.midbottom, norm_rect.center): ((norm_rect.midbottom[0], norm_rect.midbottom[1] - 1), norm_rect.center)}
        for line, expected_line in line_dict.items():
            clipped_line = rect.clipline(line)
            self.assertNotEqual(rect, norm_rect)
            self.assertTupleEqual(clipped_line, expected_line)
            expected_line = (expected_line[1], expected_line[0])
            clipped_line = rect.clipline((line[1], line[0]))
            self.assertTupleEqual(clipped_line, expected_line)