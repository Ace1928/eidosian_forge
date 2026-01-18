from functools import reduce
import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermepow(self):
    for i in range(5):
        for j in range(5):
            msg = f'At i={i}, j={j}'
            c = np.arange(i + 1)
            tgt = reduce(herme.hermemul, [c] * j, np.array([1]))
            res = herme.hermepow(c, j)
            assert_equal(trim(res), trim(tgt), err_msg=msg)