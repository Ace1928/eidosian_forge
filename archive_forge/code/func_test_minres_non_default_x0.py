import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_equal, assert_allclose, assert_
from scipy.sparse.linalg._isolve import minres
from pytest import raises as assert_raises
def test_minres_non_default_x0():
    np.random.seed(1234)
    rtol = 1e-06
    a = np.random.randn(5, 5)
    a = np.dot(a, a.T)
    b = np.random.randn(5)
    c = np.random.randn(5)
    x = minres(a, b, x0=c, rtol=rtol)[0]
    assert norm(a @ x - b) <= rtol * norm(b)