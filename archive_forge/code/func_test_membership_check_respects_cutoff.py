import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_membership_check_respects_cutoff(self):
    self.assertTrue('a' in self.vocab)
    self.assertFalse('c' in self.vocab)
    self.assertFalse('z' in self.vocab)