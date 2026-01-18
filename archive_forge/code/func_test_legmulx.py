from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_legmulx(self):
    assert_equal(leg.legmulx([0]), [0])
    assert_equal(leg.legmulx([1]), [0, 1])
    for i in range(1, 5):
        tmp = 2 * i + 1
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [i / tmp, 0, (i + 1) / tmp]
        assert_equal(leg.legmulx(ser), tgt)