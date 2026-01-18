import unittest
import pytest
from nltk import FreqDist
from nltk.lm import NgramCounter
from nltk.util import everygrams
def test_N(self):
    assert self.bigram_counter.N() == 16
    assert self.trigram_counter.N() == 21