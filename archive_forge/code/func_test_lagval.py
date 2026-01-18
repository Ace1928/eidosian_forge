from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lagval(self):
    assert_equal(lag.lagval([], [1]).size, 0)
    x = np.linspace(-1, 1)
    y = [polyval(x, c) for c in Llist]
    for i in range(7):
        msg = f'At i={i}'
        tgt = y[i]
        res = lag.lagval(x, [0] * i + [1])
        assert_almost_equal(res, tgt, err_msg=msg)
    for i in range(3):
        dims = [2] * i
        x = np.zeros(dims)
        assert_equal(lag.lagval(x, [1]).shape, dims)
        assert_equal(lag.lagval(x, [1, 0]).shape, dims)
        assert_equal(lag.lagval(x, [1, 0, 0]).shape, dims)