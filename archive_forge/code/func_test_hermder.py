from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermder(self):
    assert_raises(TypeError, herm.hermder, [0], 0.5)
    assert_raises(ValueError, herm.hermder, [0], -1)
    for i in range(5):
        tgt = [0] * i + [1]
        res = herm.hermder(tgt, m=0)
        assert_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = herm.hermder(herm.hermint(tgt, m=j), m=j)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = herm.hermder(herm.hermint(tgt, m=j, scl=2), m=j, scl=0.5)
            assert_almost_equal(trim(res), trim(tgt))