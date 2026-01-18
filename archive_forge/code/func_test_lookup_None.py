import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_lookup_None(self):
    with self.assertRaises(TypeError):
        self.vocab.lookup(None)
    with self.assertRaises(TypeError):
        list(self.vocab.lookup([None, None]))