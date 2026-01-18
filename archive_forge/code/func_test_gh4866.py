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
def test_gh4866(self):
    M = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
    R = np.array([[sqrt(0.5), 0, 0, sqrt(0.5)], [0, 0, 0, 0], [0, 0, 0, 0], [sqrt(0.5), 0, 0, sqrt(0.5)]])
    assert_allclose(np.dot(R, R), M, atol=1e-14)
    assert_allclose(sqrtm(M), R, atol=1e-14)