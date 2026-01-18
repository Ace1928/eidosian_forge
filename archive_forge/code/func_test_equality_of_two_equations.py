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
def test_equality_of_two_equations(self):
    a = array([[1, 2], [3, 4]])
    b = array([[5, 6], [7, 8]])
    res1 = khatri_rao(a, b)
    res2 = np.vstack([np.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]).T
    assert_array_equal(res1, res2)