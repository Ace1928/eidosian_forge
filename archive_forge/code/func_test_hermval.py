from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermval(self):
    assert_equal(herm.hermval([], [1]).size, 0)
    x = np.linspace(-1, 1)
    y = [polyval(x, c) for c in Hlist]
    for i in range(10):
        msg = f'At i={i}'
        tgt = y[i]
        res = herm.hermval(x, [0] * i + [1])
        assert_almost_equal(res, tgt, err_msg=msg)
    for i in range(3):
        dims = [2] * i
        x = np.zeros(dims)
        assert_equal(herm.hermval(x, [1]).shape, dims)
        assert_equal(herm.hermval(x, [1, 0]).shape, dims)
        assert_equal(herm.hermval(x, [1, 0, 0]).shape, dims)