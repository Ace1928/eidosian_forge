from functools import reduce
import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermediv(self):
    for i in range(5):
        for j in range(5):
            msg = f'At i={i}, j={j}'
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = herme.hermeadd(ci, cj)
            quo, rem = herme.hermediv(tgt, ci)
            res = herme.hermeadd(herme.hermemul(quo, ci), rem)
            assert_equal(trim(res), trim(tgt), err_msg=msg)