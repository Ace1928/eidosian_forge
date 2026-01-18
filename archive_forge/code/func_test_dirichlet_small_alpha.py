import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_dirichlet_small_alpha(self):
    eps = 1e-09
    alpha = eps * np.array([1.0, 0.001])
    random = Generator(MT19937(self.seed))
    actual = random.dirichlet(alpha, size=(3, 2))
    expected = np.array([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]])
    assert_array_almost_equal(actual, expected, decimal=15)