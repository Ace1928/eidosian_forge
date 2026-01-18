from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_gh1758(self):

    def fun(x):
        return np.sqrt(x[1])

    def f_eqcon(x):
        """ Equality constraint """
        return x[1] - (2 * x[0]) ** 3

    def f_eqcon2(x):
        """ Equality constraint """
        return x[1] - (-x[0] + 1) ** 3
    c1 = {'type': 'eq', 'fun': f_eqcon}
    c2 = {'type': 'eq', 'fun': f_eqcon2}
    res = minimize(fun, [8, 0.25], method='SLSQP', constraints=[c1, c2], bounds=[(-0.5, 1), (0, 8)])
    np.testing.assert_allclose(res.fun, 0.5443310539518)
    np.testing.assert_allclose(res.x, [0.33333333, 0.2962963])
    assert res.success