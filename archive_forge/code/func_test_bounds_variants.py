import pytest
import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose, assert_equal, assert_
from scipy.sparse import rand, coo_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import lsq_linear
from scipy.optimize._minimize import Bounds
def test_bounds_variants(self):
    x = np.array([1, 3])
    A = self.rnd.uniform(size=(2, 2))
    b = A @ x
    lb = np.array([1, 1])
    ub = np.array([2, 2])
    bounds_old = (lb, ub)
    bounds_new = Bounds(lb, ub)
    res_old = lsq_linear(A, b, bounds_old)
    res_new = lsq_linear(A, b, bounds_new)
    assert not np.allclose(res_new.x, res_new.unbounded_sol[0])
    assert_allclose(res_old.x, res_new.x)