from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lagmul(self):
    for i in range(5):
        pol1 = [0] * i + [1]
        val1 = lag.lagval(self.x, pol1)
        for j in range(5):
            msg = f'At i={i}, j={j}'
            pol2 = [0] * j + [1]
            val2 = lag.lagval(self.x, pol2)
            pol3 = lag.lagmul(pol1, pol2)
            val3 = lag.lagval(self.x, pol3)
            assert_(len(pol3) == i + j + 1, msg)
            assert_almost_equal(val3, val1 * val2, err_msg=msg)