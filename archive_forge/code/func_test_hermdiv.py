from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermdiv(self):
    for i in range(5):
        for j in range(5):
            msg = f'At i={i}, j={j}'
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = herm.hermadd(ci, cj)
            quo, rem = herm.hermdiv(tgt, ci)
            res = herm.hermadd(herm.hermmul(quo, ci), rem)
            assert_equal(trim(res), trim(tgt), err_msg=msg)