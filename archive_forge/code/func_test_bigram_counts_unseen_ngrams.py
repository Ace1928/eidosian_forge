import unittest
import pytest
from nltk import FreqDist
from nltk.lm import NgramCounter
from nltk.util import everygrams
def test_bigram_counts_unseen_ngrams(self):
    assert self.bigram_counter[['b']]['z'] == 0