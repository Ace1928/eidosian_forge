from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_gh9640(self):
    np.random.seed(10)
    cons = ({'type': 'ineq', 'fun': lambda x: -x[0] - x[1] - 3}, {'type': 'ineq', 'fun': lambda x: x[1] + x[2] - 2})
    bnds = ((-2, 2), (-2, 2), (-2, 2))

    def target(x):
        return 1
    x0 = [-1.8869783504471584, -0.640096352696244, -0.8174212253407696]
    res = minimize(target, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'disp': False, 'maxiter': 10000})
    assert not res.success