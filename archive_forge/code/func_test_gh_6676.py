from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_gh_6676(self):

    def func(x):
        return (x[0] - 1) ** 2 + 2 * (x[1] - 1) ** 2 + 0.5 * (x[2] - 1) ** 2
    sol = minimize(func, [0, 0, 0], method='SLSQP')
    assert_(sol.jac.shape == (3,))