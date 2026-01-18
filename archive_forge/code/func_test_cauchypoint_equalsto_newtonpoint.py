import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.qp_subproblem \
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import TestCase, assert_array_almost_equal, assert_equal
import pytest
def test_cauchypoint_equalsto_newtonpoint(self):
    A = np.array([[1, 8]])
    b = np.array([-16])
    _, _, Y = projections(A)
    newton_point = np.array([0.24615385, 1.96923077])
    x = modified_dogleg(A, Y, b, 2, [-np.inf, -np.inf], [np.inf, np.inf])
    assert_array_almost_equal(x, newton_point)
    x = modified_dogleg(A, Y, b, 1, [-np.inf, -np.inf], [np.inf, np.inf])
    assert_array_almost_equal(x, newton_point / np.linalg.norm(newton_point))
    x = modified_dogleg(A, Y, b, 2, [-np.inf, -np.inf], [0.1, np.inf])
    assert_array_almost_equal(x, newton_point / newton_point[0] * 0.1)