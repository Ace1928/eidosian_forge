from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_compare_one_parent(self):
    diff = multiparent.MultiParent.from_lines(LINES_1, [LINES_2])
    self.assertEqual([multiparent.ParentText(0, 0, 0, 1), multiparent.NewText([b'b\n']), multiparent.ParentText(0, 1, 2, 3)], diff.hunks)
    diff = multiparent.MultiParent.from_lines(LINES_2, [LINES_1])
    self.assertEqual([multiparent.ParentText(0, 0, 0, 1), multiparent.ParentText(0, 2, 1, 3)], diff.hunks)