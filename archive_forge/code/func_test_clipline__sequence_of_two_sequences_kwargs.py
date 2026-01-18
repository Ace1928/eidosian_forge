import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline__sequence_of_two_sequences_kwargs(self):
    """Ensures clipline handles a sequence of two sequences using kwargs.

        Tests the clipline(((x1, y1), (x2, y2))) format.
        Tests the sequences as different types.
        """
    rect = Rect((1, 2), (35, 40))
    pt1 = (5, 6)
    pt2 = (11, 19)
    INNER_SEQUENCES = (list, tuple, Vector2)
    expected_line = (pt1, pt2)
    for inner_seq1 in INNER_SEQUENCES:
        endpt1 = inner_seq1(pt1)
        for inner_seq2 in INNER_SEQUENCES:
            endpt2 = inner_seq2(pt2)
            for outer_seq in (list, tuple):
                clipped_line = rect.clipline(x1=outer_seq((endpt1, endpt2)))
                self.assertIsInstance(clipped_line, tuple)
                self.assertTupleEqual(clipped_line, expected_line)