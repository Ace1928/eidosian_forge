import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.qp_subproblem \
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import TestCase, assert_array_almost_equal, assert_equal
import pytest
def test_inactive_box_constraints(self):
    H = csc_matrix([[6, 2, 1, 3], [2, 5, 2, 4], [1, 2, 4, 5], [3, 4, 5, 7]])
    A = csc_matrix([[1, 0, 1, 0], [0, 1, 1, 1]])
    c = np.array([-2, -3, -3, 1])
    b = -np.array([3, 0])
    Z, _, Y = projections(A)
    x, info = projected_cg(H, c, Z, Y, b, tol=0, lb=[0.5, -np.inf, -np.inf, -np.inf], return_all=True)
    x_kkt, _ = eqp_kktfact(H, c, A, b)
    assert_equal(info['stop_cond'], 1)
    assert_equal(info['hits_boundary'], False)
    assert_array_almost_equal(x, x_kkt)