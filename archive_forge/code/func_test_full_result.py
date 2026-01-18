from itertools import product
import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import issparse, lil_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import least_squares, Bounds
from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES
from scipy.optimize._lsq.common import EPS, make_strictly_feasible, CL_scaling_vector
def test_full_result(self):
    res = least_squares(fun_trivial, 2.0, method=self.method)
    assert_allclose(res.x, 0, atol=0.0001)
    assert_allclose(res.cost, 12.5)
    assert_allclose(res.fun, 5)
    assert_allclose(res.jac, 0, atol=0.0001)
    assert_allclose(res.grad, 0, atol=0.01)
    assert_allclose(res.optimality, 0, atol=0.01)
    assert_equal(res.active_mask, 0)
    if self.method == 'lm':
        assert_(res.nfev < 30)
        assert_(res.njev is None)
    else:
        assert_(res.nfev < 10)
        assert_(res.njev < 10)
    assert_(res.status > 0)
    assert_(res.success)