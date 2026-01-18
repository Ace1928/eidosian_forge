import io
import unittest
from nltk.data import find
from nltk.translate.bleu_score import (
def test_length_one_hypothesis(self):
    references = ['The candidate has no alignment to any of the references'.split()]
    hypothesis = ['Foo']
    method4 = SmoothingFunction().method4
    try:
        sentence_bleu(references, hypothesis, smoothing_function=method4)
    except ValueError:
        pass