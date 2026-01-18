import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def exp_bc_jac(ya, yb):
    dbc_dya = np.array([[1, 0], [0, 0]])
    dbc_dyb = np.array([[0, 0], [1, 0]])
    return (dbc_dya, dbc_dyb)