import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def big_bc_with_parameters(ya, yb, p):
    return np.hstack((ya[::2], yb[::2], ya[1] - p[0], ya[3] - p[1]))