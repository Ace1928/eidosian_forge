from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_minimize_bounded_approximated(self):
    jacs = [None, False, '2-point', '3-point']
    for jac in jacs:
        with np.errstate(invalid='ignore'):
            res = minimize(self.fun, [-1.0, 1.0], args=(-1.0,), jac=jac, bounds=((2.5, None), (None, 0.5)), method='SLSQP', options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2.5, 0.5])
        assert_(2.5 <= res.x[0])
        assert_(res.x[1] <= 0.5)