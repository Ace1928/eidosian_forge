from numpy.testing import assert_array_almost_equal, assert_array_equal
from pytest import raises as assert_raises
from numpy import array, transpose, dot, conjugate, zeros_like, empty
from numpy.random import random
from scipy.linalg import cholesky, cholesky_banded, cho_solve_banded, \
from scipy.linalg._testutils import assert_no_overwrite
def test_cho_solve(self):
    x = array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    xcho = cho_factor(x)
    assert_no_overwrite(lambda b: cho_solve(xcho, b), [(3,)])