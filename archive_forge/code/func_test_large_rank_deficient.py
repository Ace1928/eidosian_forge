import pytest
import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose, assert_equal, assert_
from scipy.sparse import rand, coo_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import lsq_linear
from scipy.optimize._minimize import Bounds
@pytest.mark.xslow
def test_large_rank_deficient(self):
    np.random.seed(0)
    n, m = np.sort(np.random.randint(2, 1000, size=2))
    m *= 2
    A = 1.0 * np.random.randint(-99, 99, size=[m, n])
    b = 1.0 * np.random.randint(-99, 99, size=[m])
    bounds = 1.0 * np.sort(np.random.randint(-99, 99, size=(2, n)), axis=0)
    bounds[1, :] += 1.0
    w = np.random.choice(n, n)
    A = A[:, w]
    x_bvls = lsq_linear(A, b, bounds=bounds, method='bvls').x
    x_trf = lsq_linear(A, b, bounds=bounds, method='trf').x
    cost_bvls = np.sum((A @ x_bvls - b) ** 2)
    cost_trf = np.sum((A @ x_trf - b) ** 2)
    assert_(abs(cost_bvls - cost_trf) < cost_trf * 1e-10)