from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lagder(self):
    assert_raises(TypeError, lag.lagder, [0], 0.5)
    assert_raises(ValueError, lag.lagder, [0], -1)
    for i in range(5):
        tgt = [0] * i + [1]
        res = lag.lagder(tgt, m=0)
        assert_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = lag.lagder(lag.lagint(tgt, m=j), m=j)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = lag.lagder(lag.lagint(tgt, m=j, scl=2), m=j, scl=0.5)
            assert_almost_equal(trim(res), trim(tgt))