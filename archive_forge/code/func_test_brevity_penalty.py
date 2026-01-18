import io
import unittest
from nltk.data import find
from nltk.translate.bleu_score import (
def test_brevity_penalty(self):
    references = [['a'] * 11, ['a'] * 8]
    hypothesis = ['a'] * 7
    hyp_len = len(hypothesis)
    closest_ref_len = closest_ref_length(references, hyp_len)
    self.assertAlmostEqual(brevity_penalty(closest_ref_len, hyp_len), 0.8669, places=4)
    references = [['a'] * 11, ['a'] * 8, ['a'] * 6, ['a'] * 7]
    hypothesis = ['a'] * 7
    hyp_len = len(hypothesis)
    closest_ref_len = closest_ref_length(references, hyp_len)
    assert brevity_penalty(closest_ref_len, hyp_len) == 1.0