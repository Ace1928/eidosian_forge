import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def sl_bc_jac(ya, yb, p):
    dbc_dya = np.zeros((3, 2))
    dbc_dya[0, 0] = 1
    dbc_dya[2, 1] = 1
    dbc_dyb = np.zeros((3, 2))
    dbc_dyb[1, 0] = 1
    dbc_dp = np.zeros((3, 1))
    dbc_dp[2, 0] = -1
    return (dbc_dya, dbc_dyb, dbc_dp)