from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lagdiv(self):
    for i in range(5):
        for j in range(5):
            msg = f'At i={i}, j={j}'
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = lag.lagadd(ci, cj)
            quo, rem = lag.lagdiv(tgt, ci)
            res = lag.lagadd(lag.lagmul(quo, ci), rem)
            assert_almost_equal(trim(res), trim(tgt), err_msg=msg)