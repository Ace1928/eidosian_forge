import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def emden_fun_jac(x, y):
    df_dy = np.empty((2, 2, x.shape[0]))
    df_dy[0, 0] = 0
    df_dy[0, 1] = 1
    df_dy[1, 0] = -5 * y[0] ** 4
    df_dy[1, 1] = 0
    return df_dy