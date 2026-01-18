import unittest
import torch
from trl.core import masked_mean, masked_var, masked_whiten, whiten
def test_masked_whiten(self):
    whiten_unmasked = whiten(self.test_input_unmasked)
    whiten_masked = masked_whiten(self.test_input, self.test_mask)[1:3]
    diffs = (whiten_unmasked - whiten_masked).sum()
    assert abs(diffs.item()) < 1e-05