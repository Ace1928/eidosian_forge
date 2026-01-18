import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def test_modify_mesh():
    x = np.array([0, 1, 3, 9], dtype=float)
    x_new = modify_mesh(x, np.array([0]), np.array([2]))
    assert_array_equal(x_new, np.array([0, 0.5, 1, 3, 5, 7, 9]))
    x = np.array([-6, -3, 0, 3, 6], dtype=float)
    x_new = modify_mesh(x, np.array([1], dtype=int), np.array([0, 2, 3]))
    assert_array_equal(x_new, [-6, -5, -4, -3, -1.5, 0, 1, 2, 3, 4, 5, 6])