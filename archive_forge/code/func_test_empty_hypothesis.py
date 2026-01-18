import io
import unittest
from nltk.data import find
from nltk.translate.bleu_score import (
def test_empty_hypothesis(self):
    references = ['The candidate has no alignment to any of the references'.split()]
    hypothesis = []
    assert sentence_bleu(references, hypothesis) == 0