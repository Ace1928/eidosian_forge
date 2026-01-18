import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
def test_exactly_singular(self):
    A = np.array([[0, 0], [1j, 1j]])
    B = np.asarray([[1, 1], [0, 0]])
    for M in (A, A.T, B, B.T):
        expected_warning = _matfuncs_inv_ssq.LogmExactlySingularWarning
        L, info = assert_warns(expected_warning, logm, M, disp=False)
        E = expm(L)
        assert_allclose(E, M, atol=1e-14)