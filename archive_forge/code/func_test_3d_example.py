import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.qp_subproblem \
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import TestCase, assert_array_almost_equal, assert_equal
import pytest
def test_3d_example(self):
    A = np.array([[1, 8, 1], [4, 2, 2]])
    b = np.array([-16, 2])
    Z, LS, Y = projections(A)
    newton_point = np.array([-1.37090909, 2.23272727, -0.49090909])
    cauchy_point = np.array([0.11165723, 1.73068711, 0.16748585])
    origin = np.zeros_like(newton_point)
    x = modified_dogleg(A, Y, b, 3, [-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    assert_array_almost_equal(x, newton_point)
    x = modified_dogleg(A, Y, b, 2, [-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    z = cauchy_point
    d = newton_point - cauchy_point
    t = (x - z) / d
    assert_array_almost_equal(t, np.full(3, 0.4080733))
    assert_array_almost_equal(np.linalg.norm(x), 2)
    x = modified_dogleg(A, Y, b, 5, [-1, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    z = cauchy_point
    d = newton_point - cauchy_point
    t = (x - z) / d
    assert_array_almost_equal(t, np.full(3, 0.7498195))
    assert_array_almost_equal(x[0], -1)
    x = modified_dogleg(A, Y, b, 1, [-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    z = origin
    d = cauchy_point
    t = (x - z) / d
    assert_array_almost_equal(t, np.full(3, 0.573936265))
    assert_array_almost_equal(np.linalg.norm(x), 1)
    x = modified_dogleg(A, Y, b, 2, [-np.inf, -np.inf, -np.inf], [np.inf, 1, np.inf])
    z = origin
    d = newton_point
    t = (x - z) / d
    assert_array_almost_equal(t, np.full(3, 0.4478827364))
    assert_array_almost_equal(x[1], 1)