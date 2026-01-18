import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_creation_with_counter(self):
    self.assertEqual(self.vocab, Vocabulary(Counter(['z', 'a', 'b', 'c', 'f', 'd', 'e', 'g', 'a', 'd', 'b', 'e', 'w']), unk_cutoff=2))