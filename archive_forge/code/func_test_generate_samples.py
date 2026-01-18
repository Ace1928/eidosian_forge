import gc
import unittest
import torch
from trl import is_diffusers_available, is_peft_available
from .testing_utils import require_diffusers
def test_generate_samples(self):
    samples, output_pairs = self.trainer._generate_samples(1, 2)
    assert len(samples) == 1
    assert len(output_pairs) == 1
    assert len(output_pairs[0][0]) == 2