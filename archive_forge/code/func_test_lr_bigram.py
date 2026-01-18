import unittest
from nltk.metrics import (
def test_lr_bigram(self):
    self.assertAlmostEqual(BigramAssocMeasures.likelihood_ratio(2, (4, 4), 20), 2.4142743368419755, delta=_DELTA)
    self.assertAlmostEqual(BigramAssocMeasures.likelihood_ratio(1, (1, 1), 1), 0.0, delta=_DELTA)
    self.assertRaises(ValueError, BigramAssocMeasures.likelihood_ratio, *(0, (2, 2), 2))