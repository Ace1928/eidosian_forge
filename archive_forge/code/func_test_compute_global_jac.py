import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def test_compute_global_jac():
    n = 2
    m = 5
    k = 1
    i_jac, j_jac = compute_jac_indices(2, 5, 1)
    x = np.linspace(0, 1, 5)
    h = np.diff(x)
    y = np.vstack((np.sin(np.pi * x), np.pi * np.cos(np.pi * x)))
    p = np.array([3.0])
    f = sl_fun(x, y, p)
    x_middle = x[:-1] + 0.5 * h
    y_middle = 0.5 * (y[:, :-1] + y[:, 1:]) - h / 8 * (f[:, 1:] - f[:, :-1])
    df_dy, df_dp = sl_fun_jac(x, y, p)
    df_dy_middle, df_dp_middle = sl_fun_jac(x_middle, y_middle, p)
    dbc_dya, dbc_dyb, dbc_dp = sl_bc_jac(y[:, 0], y[:, -1], p)
    J = construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle, df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp)
    J = J.toarray()

    def J_block(h, p):
        return np.array([[h ** 2 * p ** 2 / 12 - 1, -0.5 * h, -h ** 2 * p ** 2 / 12 + 1, -0.5 * h], [0.5 * h * p ** 2, h ** 2 * p ** 2 / 12 - 1, 0.5 * h * p ** 2, 1 - h ** 2 * p ** 2 / 12]])
    J_true = np.zeros((m * n + k, m * n + k))
    for i in range(m - 1):
        J_true[i * n:(i + 1) * n, i * n:(i + 2) * n] = J_block(h[i], p[0])
    J_true[:(m - 1) * n:2, -1] = p * h ** 2 / 6 * (y[0, :-1] - y[0, 1:])
    J_true[1:(m - 1) * n:2, -1] = p * (h * (y[0, :-1] + y[0, 1:]) + h ** 2 / 6 * (y[1, :-1] - y[1, 1:]))
    J_true[8, 0] = 1
    J_true[9, 8] = 1
    J_true[10, 1] = 1
    J_true[10, 10] = -1
    assert_allclose(J, J_true, rtol=1e-10)
    df_dy, df_dp = estimate_fun_jac(sl_fun, x, y, p)
    df_dy_middle, df_dp_middle = estimate_fun_jac(sl_fun, x_middle, y_middle, p)
    dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(sl_bc, y[:, 0], y[:, -1], p)
    J = construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle, df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp)
    J = J.toarray()
    assert_allclose(J, J_true, rtol=2e-08, atol=2e-08)