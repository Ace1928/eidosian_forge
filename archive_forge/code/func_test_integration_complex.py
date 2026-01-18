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
def test_integration_complex():
    rtol = 0.001
    atol = 1e-06
    y0 = [0.5 + 1j]
    t_span = [0, 1]
    tc = np.linspace(t_span[0], t_span[1])
    for method, jac in product(['RK23', 'RK45', 'DOP853', 'BDF'], [None, jac_complex, jac_complex_sparse]):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'The following arguments have no effect for a chosen solver: `jac`')
            res = solve_ivp(fun_complex, t_span, y0, method=method, dense_output=True, rtol=rtol, atol=atol, jac=jac)
        assert_equal(res.t[0], t_span[0])
        assert_(res.t_events is None)
        assert_(res.y_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)
        if method == 'DOP853':
            assert res.nfev < 35
        else:
            assert res.nfev < 25
        if method == 'BDF':
            assert_equal(res.njev, 1)
            assert res.nlu < 6
        else:
            assert res.njev == 0
            assert res.nlu == 0
        y_true = sol_complex(res.t)
        e = compute_error(res.y, y_true, rtol, atol)
        assert np.all(e < 5)
        yc_true = sol_complex(tc)
        yc = res.sol(tc)
        e = compute_error(yc, yc_true, rtol, atol)
        assert np.all(e < 5)