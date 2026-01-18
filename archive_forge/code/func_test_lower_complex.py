from numpy.testing import assert_array_almost_equal, assert_array_equal
from pytest import raises as assert_raises
from numpy import array, transpose, dot, conjugate, zeros_like, empty
from numpy.random import random
from scipy.linalg import cholesky, cholesky_banded, cho_solve_banded, \
from scipy.linalg._testutils import assert_no_overwrite
def test_lower_complex(self):
    a = array([[4.0, 1.0, 0.0, 0.0], [1.0, 4.0, 0.5, 0.0], [0.0, 0.5, 4.0, -0.2j], [0.0, 0.0, 0.2j, 4.0]])
    ab = array([[4.0, 4.0, 4.0, 4.0], [1.0, 0.5, 0.2j, -1.0]])
    c = cholesky_banded(ab, lower=True)
    lfac = zeros_like(a)
    lfac[list(range(4)), list(range(4))] = c[0]
    lfac[(1, 2, 3), (0, 1, 2)] = c[1, :3]
    assert_array_almost_equal(a, dot(lfac, lfac.conj().T))
    b = array([0.0, 0.5j, 3.8j, 3.8])
    x = cho_solve_banded((c, True), b)
    assert_array_almost_equal(x, [0.0, 0.0, 1j, 1.0])