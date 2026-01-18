from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lagmulx(self):
    assert_equal(lag.lagmulx([0]), [0])
    assert_equal(lag.lagmulx([1]), [1, -1])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [-i, 2 * i + 1, -(i + 1)]
        assert_almost_equal(lag.lagmulx(ser), tgt)