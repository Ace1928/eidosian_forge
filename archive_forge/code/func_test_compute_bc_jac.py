import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def test_compute_bc_jac():
    ya = np.array([-1.0, 2])
    yb = np.array([0.5, 3])
    p = np.array([])
    dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(lambda ya, yb, p: exp_bc(ya, yb), ya, yb, p)
    dbc_dya_an, dbc_dyb_an = exp_bc_jac(ya, yb)
    assert_allclose(dbc_dya, dbc_dya_an)
    assert_allclose(dbc_dyb, dbc_dyb_an)
    assert_(dbc_dp is None)
    ya = np.array([0.0, 1])
    yb = np.array([0.0, -1])
    p = np.array([0.5])
    dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(sl_bc, ya, yb, p)
    dbc_dya_an, dbc_dyb_an, dbc_dp_an = sl_bc_jac(ya, yb, p)
    assert_allclose(dbc_dya, dbc_dya_an)
    assert_allclose(dbc_dyb, dbc_dyb_an)
    assert_allclose(dbc_dp, dbc_dp_an)
    ya = np.array([0.5, 100])
    yb = np.array([-1000, 10.5])
    p = np.array([])
    dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(lambda ya, yb, p: emden_bc(ya, yb), ya, yb, p)
    dbc_dya_an, dbc_dyb_an = emden_bc_jac(ya, yb)
    assert_allclose(dbc_dya, dbc_dya_an)
    assert_allclose(dbc_dyb, dbc_dyb_an)
    assert_(dbc_dp is None)