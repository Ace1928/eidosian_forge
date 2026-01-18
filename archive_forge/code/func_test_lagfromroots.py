from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lagfromroots(self):
    res = lag.lagfromroots([])
    assert_almost_equal(trim(res), [1])
    for i in range(1, 5):
        roots = np.cos(np.linspace(-np.pi, 0, 2 * i + 1)[1::2])
        pol = lag.lagfromroots(roots)
        res = lag.lagval(roots, pol)
        tgt = 0
        assert_(len(pol) == i + 1)
        assert_almost_equal(lag.lag2poly(pol)[-1], 1)
        assert_almost_equal(res, tgt)