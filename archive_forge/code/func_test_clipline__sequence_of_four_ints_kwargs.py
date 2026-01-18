import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline__sequence_of_four_ints_kwargs(self):
    """Ensures clipline handles a sequence of four ints using kwargs.

        Tests the clipline((x1, y1, x2, y2)) format.
        Tests the sequence as different types.
        """
    rect = Rect((1, 2), (35, 40))
    line = (5, 6, 11, 19)
    expected_line = ((line[0], line[1]), (line[2], line[3]))
    for outer_seq in (list, tuple):
        clipped_line = rect.clipline(rect_arg=outer_seq(line))
        self.assertIsInstance(clipped_line, tuple)
        self.assertTupleEqual(clipped_line, expected_line)