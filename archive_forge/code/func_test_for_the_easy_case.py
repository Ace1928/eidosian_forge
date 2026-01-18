import numpy as np
from scipy.optimize._trlib import (get_trlib_quadratic_subproblem)
from numpy.testing import (assert_,
def test_for_the_easy_case(self):
    H = np.array([[1.0, 0.0, 4.0], [0.0, 2.0, 0.0], [4.0, 0.0, 3.0]])
    g = np.array([5.0, 0.0, 4.0])
    trust_radius = 1.0
    subprob = KrylovQP(x=0, fun=lambda x: 0, jac=lambda x: g, hess=lambda x: None, hessp=lambda x, y: H.dot(y))
    p, hits_boundary = subprob.solve(trust_radius)
    assert_array_almost_equal(p, np.array([-1.0, 0.0, 0.0]))
    assert_equal(hits_boundary, True)
    assert_almost_equal(np.linalg.norm(H.dot(p) + subprob.lam * p + g), 0.0)
    assert_almost_equal(np.linalg.norm(p), trust_radius)
    trust_radius = 0.5
    p, hits_boundary = subprob.solve(trust_radius)
    assert_array_almost_equal(p, np.array([-0.46125446, 0.0, -0.19298788]))
    assert_equal(hits_boundary, True)
    assert_almost_equal(np.linalg.norm(H.dot(p) + subprob.lam * p + g), 0.0)
    assert_almost_equal(np.linalg.norm(p), trust_radius)