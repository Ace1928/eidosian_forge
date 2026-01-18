import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def big_bc_with_parameters_jac(ya, yb, p):
    n = ya.shape[0]
    dbc_dya = np.zeros((n + 2, n))
    dbc_dyb = np.zeros((n + 2, n))
    dbc_dya[range(n // 2), range(0, n, 2)] = 1
    dbc_dyb[range(n // 2, n), range(0, n, 2)] = 1
    dbc_dp = np.zeros((n + 2, 2))
    dbc_dp[n, 0] = -1
    dbc_dya[n, 1] = 1
    dbc_dp[n + 1, 1] = -1
    dbc_dya[n + 1, 3] = 1
    return (dbc_dya, dbc_dyb, dbc_dp)