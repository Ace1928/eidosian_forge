import unittest
import pytest
from nltk import FreqDist
from nltk.lm import NgramCounter
from nltk.util import everygrams
def test_ngram_conditional_freqdist(self):
    case = unittest.TestCase()
    expected_trigram_contexts = [('a', 'b'), ('b', 'c'), ('e', 'g'), ('g', 'd'), ('d', 'b')]
    expected_bigram_contexts = [('a',), ('b',), ('d',), ('e',), ('c',), ('g',)]
    bigrams = self.trigram_counter[2]
    trigrams = self.trigram_counter[3]
    self.case.assertCountEqual(expected_bigram_contexts, bigrams.conditions())
    self.case.assertCountEqual(expected_trigram_contexts, trigrams.conditions())