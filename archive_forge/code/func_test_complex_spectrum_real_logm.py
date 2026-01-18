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
def test_complex_spectrum_real_logm(self):
    M = [[1, 1, 2], [2, 1, 1], [1, 2, 1]]
    for dt in (float, complex):
        X = np.array(M, dtype=dt)
        w = scipy.linalg.eigvals(X)
        assert_(0.01 < np.absolute(w.imag).sum())
        Y, info = logm(X, disp=False)
        assert_(np.issubdtype(Y.dtype, np.inexact))
        assert_allclose(expm(Y), X)