import io
import unittest
from nltk.data import find
from nltk.translate.bleu_score import (
def test_full_matches(self):
    references = ['John loves Mary'.split()]
    hypothesis = 'John loves Mary'.split()
    for n in range(1, len(hypothesis)):
        weights = (1.0 / n,) * n
        assert sentence_bleu(references, hypothesis, weights) == 1.0