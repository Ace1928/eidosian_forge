from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_legdiv(self):
    for i in range(5):
        for j in range(5):
            msg = f'At i={i}, j={j}'
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = leg.legadd(ci, cj)
            quo, rem = leg.legdiv(tgt, ci)
            res = leg.legadd(leg.legmul(quo, ci), rem)
            assert_equal(trim(res), trim(tgt), err_msg=msg)