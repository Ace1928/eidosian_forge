import unittest
from collections import Counter
from low_index import *
def test_K15n12345_7(self):
    reps = permutation_reps(3, ['aBcACAcb'], ['aBaCacBAcAbaBabaCAcAbaBaCacBAcAbaBabCAcAbABaCabABAbABaCabCAcAb'], 7)
    degrees = Counter([len(rep[0]) for rep in reps])
    self.assertEqual(degrees[1], 1)
    self.assertEqual(degrees[2], 1)
    self.assertEqual(degrees[3], 1)
    self.assertEqual(degrees[4], 1)
    self.assertEqual(degrees[5], 3)
    self.assertEqual(degrees[6], 11)
    self.assertEqual(degrees[7], 22)
    self.assertIn([[0], [0], [0]], reps)
    self.assertIn([[0, 2, 1, 3, 4, 6, 5], [0, 3, 4, 5, 1, 2, 6], [1, 2, 0, 5, 6, 3, 4]], reps)