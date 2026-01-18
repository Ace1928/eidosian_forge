import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.qp_subproblem \
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import TestCase, assert_array_almost_equal, assert_equal
import pytest
def test_negative_curvature(self):
    H = csc_matrix([[1, 2, 1, 3], [2, 0, 2, 4], [1, 2, 0, 2], [3, 4, 2, 0]])
    A = csc_matrix([[1, 0, 1, 0], [0, 1, 0, 1]])
    c = np.array([-2, -3, -3, 1])
    b = -np.array([3, 0])
    Z, _, Y = projections(A)
    trust_radius = 1000
    x, info = projected_cg(H, c, Z, Y, b, tol=0, trust_radius=trust_radius)
    assert_equal(info['stop_cond'], 3)
    assert_equal(info['hits_boundary'], True)
    assert_array_almost_equal(np.linalg.norm(x), trust_radius)