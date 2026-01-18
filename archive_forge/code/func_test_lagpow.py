from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lagpow(self):
    for i in range(5):
        for j in range(5):
            msg = f'At i={i}, j={j}'
            c = np.arange(i + 1)
            tgt = reduce(lag.lagmul, [c] * j, np.array([1]))
            res = lag.lagpow(c, j)
            assert_equal(trim(res), trim(tgt), err_msg=msg)