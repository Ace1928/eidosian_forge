from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_minimize_bounded_constraint(self):

    def c(x):
        assert 0 <= x[0] <= 1 and 0 <= x[1] <= 1, x
        return x[0] ** 0.5 + x[1]

    def f(x):
        assert 0 <= x[0] <= 1 and 0 <= x[1] <= 1, x
        return -x[0] ** 2 + x[1] ** 2
    cns = [NonlinearConstraint(c, 0, 1.5)]
    x0 = np.asarray([0.9, 0.5])
    bnd = Bounds([0.0, 0.0], [1.0, 1.0])
    minimize(f, x0, method='SLSQP', bounds=bnd, constraints=cns)