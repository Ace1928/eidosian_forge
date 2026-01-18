import unittest
import pytest
from nltk import FreqDist
from nltk.lm import NgramCounter
from nltk.util import everygrams
def test_bigram_counts_seen_ngrams(self):
    assert self.bigram_counter[['a']]['b'] == 1
    assert self.bigram_counter[['b']]['c'] == 1