import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.slow
def test_dirichlet_moderately_small_alpha(self):
    alpha = np.array([0.02, 0.04, 0.03])
    exact_mean = alpha / alpha.sum()
    random = Generator(MT19937(self.seed))
    sample = random.dirichlet(alpha, size=20000000)
    sample_mean = sample.mean(axis=0)
    assert_allclose(sample_mean, exact_mean, rtol=0.001)