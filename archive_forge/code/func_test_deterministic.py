import pytest
import numpy as np
from scipy.optimize import quadratic_assignment, OptimizeWarning
from scipy.optimize._qap import _calc_score as _score
from numpy.testing import assert_equal, assert_, assert_warns
def test_deterministic(self):
    n = 20
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    res1 = quadratic_assignment(A, B, method=self.method)
    np.random.seed(0)
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    res2 = quadratic_assignment(A, B, method=self.method)
    assert_equal(res1.nit, res2.nit)