import pytest
import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose, assert_equal, assert_
from scipy.sparse import rand, coo_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import lsq_linear
from scipy.optimize._minimize import Bounds
def test_dense_rank_deficient(self):
    A = np.array([[-0.307, -0.184]])
    b = np.array([0.773])
    lb = [-0.1, -0.1]
    ub = [0.1, 0.1]
    for lsq_solver in self.lsq_solvers:
        res = lsq_linear(A, b, (lb, ub), method=self.method, lsq_solver=lsq_solver)
        assert_allclose(res.x, [-0.1, -0.1])
        assert_allclose(res.unbounded_sol[0], lstsq(A, b, rcond=-1)[0])
    A = np.array([[0.334, 0.668], [-0.516, -1.032], [0.192, 0.384]])
    b = np.array([-1.436, 0.135, 0.909])
    lb = [0, -1]
    ub = [1, -0.5]
    for lsq_solver in self.lsq_solvers:
        res = lsq_linear(A, b, (lb, ub), method=self.method, lsq_solver=lsq_solver)
        assert_allclose(res.optimality, 0, atol=1e-11)
        assert_allclose(res.unbounded_sol[0], lstsq(A, b, rcond=-1)[0])