from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebmul(self):
    for i in range(5):
        for j in range(5):
            msg = f'At i={i}, j={j}'
            tgt = np.zeros(i + j + 1)
            tgt[i + j] += 0.5
            tgt[abs(i - j)] += 0.5
            res = cheb.chebmul([0] * i + [1], [0] * j + [1])
            assert_equal(trim(res), trim(tgt), err_msg=msg)