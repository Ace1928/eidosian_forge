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
def test_bounds_instances(self):
    res = least_squares(fun_trivial, 0.5, bounds=Bounds())
    assert_allclose(res.x, 0.0, atol=0.0001)
    res = least_squares(fun_trivial, 3.0, bounds=Bounds(lb=1.0))
    assert_allclose(res.x, 1.0, atol=0.0001)
    res = least_squares(fun_trivial, 0.5, bounds=Bounds(lb=-1.0, ub=1.0))
    assert_allclose(res.x, 0.0, atol=0.0001)
    res = least_squares(fun_trivial, -3.0, bounds=Bounds(ub=-1.0))
    assert_allclose(res.x, -1.0, atol=0.0001)
    res = least_squares(fun_2d_trivial, [0.5, 0.5], bounds=Bounds(lb=[-1.0, -1.0], ub=1.0))
    assert_allclose(res.x, [0.0, 0.0], atol=1e-05)
    res = least_squares(fun_2d_trivial, [0.5, 0.5], bounds=Bounds(lb=[0.1, 0.1]))
    assert_allclose(res.x, [0.1, 0.1], atol=1e-05)