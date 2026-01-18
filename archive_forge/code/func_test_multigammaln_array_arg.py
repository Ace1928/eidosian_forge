import numpy as np
from numpy.testing import (assert_array_equal,
from pytest import raises as assert_raises
from scipy.special import gammaln, multigammaln
def test_multigammaln_array_arg():
    np.random.seed(1234)
    cases = [(np.abs(np.random.randn(3, 2)) + 5, 5), (np.abs(np.random.randn(1, 2)) + 5, 5), (np.arange(10.0, 18.0).reshape(2, 2, 2), 3), (np.array([2.0]), 3), (np.float64(2.0), 3)]
    for a, d in cases:
        _check_multigammaln_array_result(a, d)