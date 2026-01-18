import numpy as np
from scipy.optimize._trlib import (get_trlib_quadratic_subproblem)
from numpy.testing import (assert_,
def test_disp(self, capsys):
    H = -np.eye(5)
    g = np.array([0, 0, 0, 0, 1e-06])
    trust_radius = 1.1
    subprob = KrylovQP_disp(x=0, fun=lambda x: 0, jac=lambda x: g, hess=lambda x: None, hessp=lambda x, y: H.dot(y))
    p, hits_boundary = subprob.solve(trust_radius)
    out, err = capsys.readouterr()
    assert_(out.startswith(' TR Solving trust region problem'), repr(out))