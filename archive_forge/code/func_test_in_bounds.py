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
def test_in_bounds(self):
    for jac in ['2-point', '3-point', 'cs', jac_trivial]:
        res = least_squares(fun_trivial, 2.0, jac=jac, bounds=(-1.0, 3.0), method=self.method)
        assert_allclose(res.x, 0.0, atol=0.0001)
        assert_equal(res.active_mask, [0])
        assert_(-1 <= res.x <= 3)
        res = least_squares(fun_trivial, 2.0, jac=jac, bounds=(0.5, 3.0), method=self.method)
        assert_allclose(res.x, 0.5, atol=0.0001)
        assert_equal(res.active_mask, [-1])
        assert_(0.5 <= res.x <= 3)