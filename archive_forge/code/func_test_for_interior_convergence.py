import numpy as np
from scipy.optimize._trlib import (get_trlib_quadratic_subproblem)
from numpy.testing import (assert_,
def test_for_interior_convergence(self):
    H = np.array([[1.812159, 0.82687265, 0.21838879, -0.52487006, 0.25436988], [0.82687265, 2.66380283, 0.31508988, -0.40144163, 0.08811588], [0.21838879, 0.31508988, 2.38020726, -0.3166346, 0.27363867], [-0.52487006, -0.40144163, -0.3166346, 1.61927182, -0.42140166], [0.25436988, 0.08811588, 0.27363867, -0.42140166, 1.33243101]])
    g = np.array([0.75798952, 0.01421945, 0.33847612, 0.83725004, -0.47909534])
    trust_radius = 1.1
    subprob = KrylovQP(x=0, fun=lambda x: 0, jac=lambda x: g, hess=lambda x: None, hessp=lambda x, y: H.dot(y))
    p, hits_boundary = subprob.solve(trust_radius)
    assert_almost_equal(np.linalg.norm(H.dot(p) + subprob.lam * p + g), 0.0)
    assert_array_almost_equal(p, [-0.68585435, 0.1222621, -0.22090999, -0.67005053, 0.31586769])
    assert_array_almost_equal(hits_boundary, False)