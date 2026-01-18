import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
@pytest.mark.parametrize('du,d,dl,b,x', [(np.array([2.1, -1.0, 1.9, 8.0]), np.array([3.0, 2.3, -5.0, -0.9, 7.1]), np.array([3.4, 3.6, 7.0, -6.0]), np.array([[2.7, 6.6], [-0.5, 10.8], [2.6, -3.2], [0.6, -11.2], [2.7, 19.1]]), np.array([[-4, 5], [7, -4], [3, -3], [-4, -2], [-3, 1]])), (np.array([2 - 1j, 2 + 1j, -1 + 1j, 1 - 1j]), np.array([-1.3 + 1.3j, -1.3 + 1.3j, -1.3 + 3.3j, -0.3 + 4.3j, -3.3 + 1.3j]), np.array([1 - 2j, 1 + 1j, 2 - 3j, 1 + 1j]), np.array([[2.4 - 5j, 2.7 + 6.9j], [3.4 + 18.2j, -6.9 - 5.3j], [-14.7 + 9.7j, -6 - 0.6j], [31.9 - 7.7j, -3.9 + 9.3j], [-1 + 1.6j, -3 + 12.2j]]), np.array([[1 + 1j, 2 - 1j], [3 - 1j, 1 + 2j], [4 + 5j, -1 + 1j], [-1 - 2j, 2 + 1j], [1 - 1j, 2 - 2j]]))])
def test_gtsvx_NAG(du, d, dl, b, x):
    gtsvx = get_lapack_funcs('gtsvx', dtype=d.dtype)
    gtsvx_out = gtsvx(dl, d, du, b)
    dlf, df, duf, du2f, ipiv, x_soln, rcond, ferr, berr, info = gtsvx_out
    assert_array_almost_equal(x, x_soln)