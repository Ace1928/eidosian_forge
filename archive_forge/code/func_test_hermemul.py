from functools import reduce
import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermemul(self):
    for i in range(5):
        pol1 = [0] * i + [1]
        val1 = herme.hermeval(self.x, pol1)
        for j in range(5):
            msg = f'At i={i}, j={j}'
            pol2 = [0] * j + [1]
            val2 = herme.hermeval(self.x, pol2)
            pol3 = herme.hermemul(pol1, pol2)
            val3 = herme.hermeval(self.x, pol3)
            assert_(len(pol3) == i + j + 1, msg)
            assert_almost_equal(val3, val1 * val2, err_msg=msg)