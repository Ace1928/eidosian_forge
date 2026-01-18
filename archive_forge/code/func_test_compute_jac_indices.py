import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def test_compute_jac_indices():
    n = 2
    m = 4
    k = 2
    i, j = compute_jac_indices(n, m, k)
    s = coo_matrix((np.ones_like(i), (i, j))).toarray()
    s_true = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 1, 1, 1, 1]])
    assert_array_equal(s, s_true)