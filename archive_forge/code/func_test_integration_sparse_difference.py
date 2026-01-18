from itertools import product
from numpy.testing import (assert_, assert_allclose, assert_array_less,
import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._numdiff import group_columns
from scipy.integrate import solve_ivp, RK23, RK45, DOP853, Radau, BDF, LSODA
from scipy.integrate import OdeSolution
from scipy.integrate._ivp.common import num_jac
from scipy.integrate._ivp.base import ConstantDenseOutput
from scipy.sparse import coo_matrix, csc_matrix
def test_integration_sparse_difference():
    n = 200
    t_span = [0, 20]
    y0 = np.zeros(2 * n)
    y0[1::2] = 1
    sparsity = medazko_sparsity(n)
    for method in ['BDF', 'Radau']:
        res = solve_ivp(fun_medazko, t_span, y0, method=method, jac_sparsity=sparsity)
        assert_equal(res.t[0], t_span[0])
        assert_(res.t_events is None)
        assert_(res.y_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)
        assert_allclose(res.y[78, -1], 0.000233994, rtol=0.01)
        assert_allclose(res.y[79, -1], 0, atol=0.001)
        assert_allclose(res.y[148, -1], 0.000359561, rtol=0.01)
        assert_allclose(res.y[149, -1], 0, atol=0.001)
        assert_allclose(res.y[198, -1], 0.000117374129, rtol=0.01)
        assert_allclose(res.y[199, -1], 6.190807e-06, atol=0.001)
        assert_allclose(res.y[238, -1], 0, atol=0.001)
        assert_allclose(res.y[239, -1], 0.9999997, rtol=0.01)