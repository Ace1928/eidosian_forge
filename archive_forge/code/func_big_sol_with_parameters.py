import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def big_sol_with_parameters(x, p):
    return np.vstack((np.sin(p[0] * x), np.sin(p[1] * x)))