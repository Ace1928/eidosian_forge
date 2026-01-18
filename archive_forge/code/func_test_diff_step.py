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
def test_diff_step(self):
    res1 = least_squares(fun_trivial, 2.0, diff_step=0.1, method=self.method)
    res2 = least_squares(fun_trivial, 2.0, diff_step=-0.1, method=self.method)
    res3 = least_squares(fun_trivial, 2.0, diff_step=None, method=self.method)
    assert_allclose(res1.x, 0, atol=0.0001)
    assert_allclose(res2.x, 0, atol=0.0001)
    assert_allclose(res3.x, 0, atol=0.0001)
    assert_equal(res1.x, res2.x)
    assert_equal(res1.nfev, res2.nfev)