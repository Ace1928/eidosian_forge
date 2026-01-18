import unittest
from collections import Counter
from low_index import *
def test_figure_eight(self):
    t = SimsTree(2, 6, [[1, 1, 1, 2, -1, -2, -2, -1, 2], [2, 1, 1, 1, 2, -1, -2, -2, -1], [-1, 2, 1, 1, 1, 2, -1, -2, -2], [-2, -1, 2, 1, 1, 1, 2, -1, -2], [-2, -2, -1, 2, 1, 1, 1, 2, -1], [-1, -2, -2, -1, 2, 1, 1, 1, 2], [2, -1, -2, -2, -1, 2, 1, 1, 1], [1, 2, -1, -2, -2, -1, 2, 1, 1], [1, 1, 2, -1, -2, -2, -1, 2, 1], [1, 1, 1, 2, -1, -2, -2, -1, 2]], [])
    degrees = Counter([cover.degree for cover in t.list()])
    self.assertEqual(degrees[0], 0)
    self.assertEqual(degrees[1], 1)
    self.assertEqual(degrees[2], 1)
    self.assertEqual(degrees[3], 1)
    self.assertEqual(degrees[4], 2)
    self.assertEqual(degrees[5], 4)
    self.assertEqual(degrees[6], 11)
    self.assertEqual(degrees[7], 0)