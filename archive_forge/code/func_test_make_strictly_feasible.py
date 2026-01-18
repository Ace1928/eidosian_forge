from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._lsq.common import (
def test_make_strictly_feasible(self):
    lb = np.array([-0.5, -0.8, 2.0])
    ub = np.array([0.8, 1.0, 3.0])
    x = np.array([-0.5, 0.0, 2 + 1e-10])
    x_new = make_strictly_feasible(x, lb, ub, rstep=0)
    assert_(x_new[0] > -0.5)
    assert_equal(x_new[1:], x[1:])
    x_new = make_strictly_feasible(x, lb, ub, rstep=0.0001)
    assert_equal(x_new, [-0.5 + 0.0001, 0.0, 2 * (1 + 0.0001)])
    x = np.array([-0.5, -1, 3.1])
    x_new = make_strictly_feasible(x, lb, ub)
    assert_(np.all((x_new >= lb) & (x_new <= ub)))
    x_new = make_strictly_feasible(x, lb, ub, rstep=0)
    assert_(np.all((x_new >= lb) & (x_new <= ub)))
    lb = np.array([-1, 100.0])
    ub = np.array([1, 100.0 + 1e-10])
    x = np.array([0, 100.0])
    x_new = make_strictly_feasible(x, lb, ub, rstep=1e-08)
    assert_equal(x_new, [0, 100.0 + 5e-11])