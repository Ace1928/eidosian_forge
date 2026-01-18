from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_bound_equality_given2(self):
    res = fmin_slsqp(self.fun, [-1.0, 1.0], fprime=self.jac, args=(-1.0,), bounds=[(-0.8, 1.0), (-1, 0.8)], f_eqcons=self.f_eqcon, fprime_eqcons=self.fprime_eqcon, iprint=0, full_output=1)
    x, fx, its, imode, smode = res
    assert_(imode == 0, imode)
    assert_array_almost_equal(x, [0.8, 0.8], decimal=3)
    assert_(-0.8 <= x[0] <= 1)
    assert_(-1 <= x[1] <= 0.8)