from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_legval(self):
    assert_equal(leg.legval([], [1]).size, 0)
    x = np.linspace(-1, 1)
    y = [polyval(x, c) for c in Llist]
    for i in range(10):
        msg = f'At i={i}'
        tgt = y[i]
        res = leg.legval(x, [0] * i + [1])
        assert_almost_equal(res, tgt, err_msg=msg)
    for i in range(3):
        dims = [2] * i
        x = np.zeros(dims)
        assert_equal(leg.legval(x, [1]).shape, dims)
        assert_equal(leg.legval(x, [1, 0]).shape, dims)
        assert_equal(leg.legval(x, [1, 0, 0]).shape, dims)