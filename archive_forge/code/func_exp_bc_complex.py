import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def exp_bc_complex(ya, yb):
    return np.hstack((ya[0] - 1 - 1j, yb[0]))