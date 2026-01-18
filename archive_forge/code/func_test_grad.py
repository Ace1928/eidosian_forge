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
def test_grad(self):
    x = np.array([2.0])
    res = least_squares(fun_trivial, x, jac_trivial, loss='linear', max_nfev=1, method=self.method)
    assert_equal(res.grad, 2 * x * (x ** 2 + 5))
    res = least_squares(fun_trivial, x, jac_trivial, loss='huber', max_nfev=1, method=self.method)
    assert_equal(res.grad, 2 * x)
    res = least_squares(fun_trivial, x, jac_trivial, loss='soft_l1', max_nfev=1, method=self.method)
    assert_allclose(res.grad, 2 * x * (x ** 2 + 5) / (1 + (x ** 2 + 5) ** 2) ** 0.5)
    res = least_squares(fun_trivial, x, jac_trivial, loss='cauchy', max_nfev=1, method=self.method)
    assert_allclose(res.grad, 2 * x * (x ** 2 + 5) / (1 + (x ** 2 + 5) ** 2))
    res = least_squares(fun_trivial, x, jac_trivial, loss='arctan', max_nfev=1, method=self.method)
    assert_allclose(res.grad, 2 * x * (x ** 2 + 5) / (1 + (x ** 2 + 5) ** 4))
    res = least_squares(fun_trivial, x, jac_trivial, loss=cubic_soft_l1, max_nfev=1, method=self.method)
    assert_allclose(res.grad, 2 * x * (x ** 2 + 5) / (1 + (x ** 2 + 5) ** 2) ** (2 / 3))