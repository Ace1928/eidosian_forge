from numpy.testing import assert_array_almost_equal, assert_array_equal
from pytest import raises as assert_raises
from numpy import array, transpose, dot, conjugate, zeros_like, empty
from numpy.random import random
from scipy.linalg import cholesky, cholesky_banded, cho_solve_banded, \
from scipy.linalg._testutils import assert_no_overwrite
def test_cho_factor_empty_square(self):
    a = empty((0, 0))
    b = array([])
    c = array([[]])
    d = []
    e = [[]]
    x, _ = cho_factor(a)
    assert_array_equal(x, a)
    for x in [b, c, d, e]:
        assert_raises(ValueError, cho_factor, x)