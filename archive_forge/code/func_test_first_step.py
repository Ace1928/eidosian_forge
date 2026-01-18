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
def test_first_step():
    rtol = 0.001
    atol = 1e-06
    y0 = [1 / 3, 2 / 9]
    first_step = 0.1
    for method in [RK23, RK45, DOP853, Radau, BDF, LSODA]:
        for t_span in ([5, 9], [5, 1]):
            res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, max_step=0.5, atol=atol, method=method, dense_output=True, first_step=first_step)
            assert_equal(res.t[0], t_span[0])
            assert_equal(res.t[-1], t_span[-1])
            assert_allclose(first_step, np.abs(res.t[1] - 5))
            assert_(res.t_events is None)
            assert_(res.success)
            assert_equal(res.status, 0)
            y_true = sol_rational(res.t)
            e = compute_error(res.y, y_true, rtol, atol)
            assert_(np.all(e < 5))
            tc = np.linspace(*t_span)
            yc_true = sol_rational(tc)
            yc = res.sol(tc)
            e = compute_error(yc, yc_true, rtol, atol)
            assert_(np.all(e < 5))
            assert_allclose(res.sol(res.t), res.y, rtol=1e-15, atol=1e-15)
            assert_raises(ValueError, method, fun_rational, t_span[0], y0, t_span[1], first_step=-1)
            assert_raises(ValueError, method, fun_rational, t_span[0], y0, t_span[1], first_step=5)