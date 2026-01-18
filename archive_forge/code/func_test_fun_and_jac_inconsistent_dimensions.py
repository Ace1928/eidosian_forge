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
def test_fun_and_jac_inconsistent_dimensions(self):
    x0 = [1, 2]
    assert_raises(ValueError, least_squares, fun_rosenbrock, x0, jac_rosenbrock_bad_dim, method=self.method)