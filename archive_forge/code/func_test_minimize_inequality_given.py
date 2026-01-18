from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_minimize_inequality_given(self):
    res = minimize(self.fun, [-1.0, 1.0], method='SLSQP', jac=self.jac, args=(-1.0,), constraints={'type': 'ineq', 'fun': self.f_ieqcon, 'args': (-1.0,)}, options=self.opts)
    assert_(res['success'], res['message'])
    assert_allclose(res.x, [2, 1], atol=0.001)