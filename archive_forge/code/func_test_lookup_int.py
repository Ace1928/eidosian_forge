import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_lookup_int(self):
    with self.assertRaises(TypeError):
        self.vocab.lookup(1)
    with self.assertRaises(TypeError):
        list(self.vocab.lookup([1, 2]))