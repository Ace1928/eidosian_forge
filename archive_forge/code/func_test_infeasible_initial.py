from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_infeasible_initial(self):

    def f(x):
        x, = x
        return x * x - 2 * x + 1
    cons_u = [{'type': 'ineq', 'fun': lambda x: 0 - x}]
    cons_l = [{'type': 'ineq', 'fun': lambda x: x - 2}]
    cons_ul = [{'type': 'ineq', 'fun': lambda x: 0 - x}, {'type': 'ineq', 'fun': lambda x: x + 1}]
    sol = minimize(f, [10], method='slsqp', constraints=cons_u)
    assert_(sol.success)
    assert_allclose(sol.x, 0, atol=1e-10)
    sol = minimize(f, [-10], method='slsqp', constraints=cons_l)
    assert_(sol.success)
    assert_allclose(sol.x, 2, atol=1e-10)
    sol = minimize(f, [-10], method='slsqp', constraints=cons_u)
    assert_(sol.success)
    assert_allclose(sol.x, 0, atol=1e-10)
    sol = minimize(f, [10], method='slsqp', constraints=cons_l)
    assert_(sol.success)
    assert_allclose(sol.x, 2, atol=1e-10)
    sol = minimize(f, [-0.5], method='slsqp', constraints=cons_ul)
    assert_(sol.success)
    assert_allclose(sol.x, 0, atol=1e-10)
    sol = minimize(f, [10], method='slsqp', constraints=cons_ul)
    assert_(sol.success)
    assert_allclose(sol.x, 0, atol=1e-10)