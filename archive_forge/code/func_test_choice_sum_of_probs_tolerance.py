import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_choice_sum_of_probs_tolerance(self):
    random.seed(1234)
    a = [1, 2, 3]
    counts = [4, 4, 2]
    for dt in (np.float16, np.float32, np.float64):
        probs = np.array(counts, dtype=dt) / sum(counts)
        c = random.choice(a, p=probs)
        assert_(c in a)
        assert_raises(ValueError, random.choice, a, p=probs * 0.9)