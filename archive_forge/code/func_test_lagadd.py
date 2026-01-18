from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lagadd(self):
    for i in range(5):
        for j in range(5):
            msg = f'At i={i}, j={j}'
            tgt = np.zeros(max(i, j) + 1)
            tgt[i] += 1
            tgt[j] += 1
            res = lag.lagadd([0] * i + [1], [0] * j + [1])
            assert_equal(trim(res), trim(tgt), err_msg=msg)