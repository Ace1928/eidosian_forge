import pytest
import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose, assert_equal, assert_
from scipy.sparse import rand, coo_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import lsq_linear
from scipy.optimize._minimize import Bounds
def test_dense_no_bounds(self):
    for lsq_solver in self.lsq_solvers:
        res = lsq_linear(A, b, method=self.method, lsq_solver=lsq_solver)
        assert_allclose(res.x, lstsq(A, b, rcond=-1)[0])
        assert_allclose(res.x, res.unbounded_sol[0])