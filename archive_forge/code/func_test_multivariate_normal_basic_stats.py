import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.parametrize('method', ['svd', 'eigh', 'cholesky'])
def test_multivariate_normal_basic_stats(self, method):
    random = Generator(MT19937(self.seed))
    n_s = 1000
    mean = np.array([1, 2])
    cov = np.array([[2, 1], [1, 2]])
    s = random.multivariate_normal(mean, cov, size=(n_s,), method=method)
    s_center = s - mean
    cov_emp = s_center.T @ s_center / (n_s - 1)
    assert np.all(np.abs(s_center.mean(-2)) < 0.1)
    assert np.all(np.abs(cov_emp - cov) < 0.2)