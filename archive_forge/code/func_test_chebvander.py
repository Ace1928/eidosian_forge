from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebvander(self):
    x = np.arange(3)
    v = cheb.chebvander(x, 3)
    assert_(v.shape == (3, 4))
    for i in range(4):
        coef = [0] * i + [1]
        assert_almost_equal(v[..., i], cheb.chebval(x, coef))
    x = np.array([[1, 2], [3, 4], [5, 6]])
    v = cheb.chebvander(x, 3)
    assert_(v.shape == (3, 2, 4))
    for i in range(4):
        coef = [0] * i + [1]
        assert_almost_equal(v[..., i], cheb.chebval(x, coef))