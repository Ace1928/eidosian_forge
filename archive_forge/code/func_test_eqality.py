import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_eqality(self):
    v1 = Vocabulary(['a', 'b', 'c'], unk_cutoff=1)
    v2 = Vocabulary(['a', 'b', 'c'], unk_cutoff=1)
    v3 = Vocabulary(['a', 'b', 'c'], unk_cutoff=1, unk_label='blah')
    v4 = Vocabulary(['a', 'b'], unk_cutoff=1)
    self.assertEqual(v1, v2)
    self.assertNotEqual(v1, v3)
    self.assertNotEqual(v1, v4)