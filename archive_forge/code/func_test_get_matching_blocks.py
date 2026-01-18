from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_get_matching_blocks(self):
    diff = multiparent.MultiParent.from_lines(LINES_1, [LINES_2])
    self.assertEqual([(0, 0, 1), (1, 2, 3), (4, 5, 0)], list(diff.get_matching_blocks(0, len(LINES_2))))
    diff = multiparent.MultiParent.from_lines(LINES_2, [LINES_1])
    self.assertEqual([(0, 0, 1), (2, 1, 3), (5, 4, 0)], list(diff.get_matching_blocks(0, len(LINES_1))))