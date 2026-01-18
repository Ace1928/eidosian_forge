import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_lookup_empty_iterables(self):
    self.assertEqual(self.vocab.lookup(()), ())
    self.assertEqual(self.vocab.lookup([]), ())
    self.assertEqual(self.vocab.lookup(iter([])), ())
    self.assertEqual(self.vocab.lookup((n for n in range(0, 0))), ())