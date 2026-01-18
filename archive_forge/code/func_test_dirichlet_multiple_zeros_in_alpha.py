import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.parametrize('alpha', [[5, 9, 0, 8], [0.5, 0, 0, 0], [1, 5, 0, 0, 1.5, 0, 0, 0], [0.01, 0.03, 0, 0.005], [1e-05, 0, 0, 0], [0.002, 0.015, 0, 0, 0.04, 0, 0, 0], [0.0], [0, 0, 0]])
def test_dirichlet_multiple_zeros_in_alpha(self, alpha):
    alpha = np.array(alpha)
    y = random.dirichlet(alpha)
    assert_equal(y[alpha == 0], 0.0)