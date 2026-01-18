import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
def test_empty_constraint(self):

    def function(x):
        return x[0] ** 2 + x[1] ** 2

    def functionjacobian(x):
        return np.array([2.0 * x[0], 2.0 * x[1]])

    def functionhvp(x, v):
        return 2.0 * v

    def constraint(x):
        return np.array([x[0] ** 2 - x[1] ** 2])

    def constraintjacobian(x):
        return np.array([[2 * x[0], -2 * x[1]]])

    def constraintlcoh(x, v):
        return np.array([[2.0, 0.0], [0.0, -2.0]]) * v[0]
    constraint = NonlinearConstraint(constraint, 1.0, np.inf, constraintjacobian, constraintlcoh)
    startpoint = [1.0, 2.0]
    bounds = Bounds([-np.inf, -np.inf], [np.inf, np.inf])
    result = minimize(function, startpoint, method='trust-constr', jac=functionjacobian, hessp=functionhvp, constraints=[constraint], bounds=bounds)
    assert_array_almost_equal(abs(result.x), np.array([1, 0]), decimal=4)