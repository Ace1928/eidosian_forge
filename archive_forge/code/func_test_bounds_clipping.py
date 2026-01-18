from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_bounds_clipping(self):

    def f(x):
        return (x[0] - 1) ** 2
    sol = minimize(f, [10], method='slsqp', bounds=[(None, 0)])
    assert_(sol.success)
    assert_allclose(sol.x, 0, atol=1e-10)
    sol = minimize(f, [-10], method='slsqp', bounds=[(2, None)])
    assert_(sol.success)
    assert_allclose(sol.x, 2, atol=1e-10)
    sol = minimize(f, [-10], method='slsqp', bounds=[(None, 0)])
    assert_(sol.success)
    assert_allclose(sol.x, 0, atol=1e-10)
    sol = minimize(f, [10], method='slsqp', bounds=[(2, None)])
    assert_(sol.success)
    assert_allclose(sol.x, 2, atol=1e-10)
    sol = minimize(f, [-0.5], method='slsqp', bounds=[(-1, 0)])
    assert_(sol.success)
    assert_allclose(sol.x, 0, atol=1e-10)
    sol = minimize(f, [10], method='slsqp', bounds=[(-1, 0)])
    assert_(sol.success)
    assert_allclose(sol.x, 0, atol=1e-10)