from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermpow(self):
    for i in range(5):
        for j in range(5):
            msg = f'At i={i}, j={j}'
            c = np.arange(i + 1)
            tgt = reduce(herm.hermmul, [c] * j, np.array([1]))
            res = herm.hermpow(c, j)
            assert_equal(trim(res), trim(tgt), err_msg=msg)