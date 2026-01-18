import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_lookup_iterables(self):
    self.assertEqual(self.vocab.lookup(['a', 'b']), ('a', 'b'))
    self.assertEqual(self.vocab.lookup(('a', 'b')), ('a', 'b'))
    self.assertEqual(self.vocab.lookup(('a', 'c')), ('a', '<UNK>'))
    self.assertEqual(self.vocab.lookup(map(str, range(3))), ('<UNK>', '<UNK>', '<UNK>'))