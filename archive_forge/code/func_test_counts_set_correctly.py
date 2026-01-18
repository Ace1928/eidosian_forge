import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_counts_set_correctly(self):
    self.assertEqual(self.vocab.counts['a'], 2)
    self.assertEqual(self.vocab.counts['b'], 2)
    self.assertEqual(self.vocab.counts['c'], 1)