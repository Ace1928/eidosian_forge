import math
import numpy as np
from numpy import array, eye, exp, random
from numpy.testing import (
from scipy.sparse import csc_matrix, csc_array, SparseEfficiencyWarning
from scipy.sparse._construct import eye as speye
from scipy.sparse.linalg._matfuncs import (expm, _expm,
from scipy.sparse._sputils import matrix
from scipy.linalg import logm
from scipy.special import factorial, binom
import scipy.sparse
import scipy.sparse.linalg
def test_burkardt_13(self):
    A4_actual = _burkardt_13_power(4, 1)
    A4_desired = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0.0001, 0, 0, 0]]
    assert_allclose(A4_actual, A4_desired)
    for n in (2, 3, 4, 10):
        k = max(1, int(np.ceil(16 / n)))
        desired = np.zeros((n, n), dtype=float)
        for p in range(n * k):
            Ap = _burkardt_13_power(n, p)
            assert_equal(np.min(Ap), 0)
            assert_allclose(np.max(Ap), np.power(10, -np.floor(p / n) * n))
            desired += Ap / factorial(p)
        actual = expm(_burkardt_13_power(n, 1))
        assert_allclose(actual, desired)