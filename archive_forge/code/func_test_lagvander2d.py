from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lagvander2d(self):
    x1, x2, x3 = self.x
    c = np.random.random((2, 3))
    van = lag.lagvander2d(x1, x2, [1, 2])
    tgt = lag.lagval2d(x1, x2, c)
    res = np.dot(van, c.flat)
    assert_almost_equal(res, tgt)
    van = lag.lagvander2d([x1], [x2], [1, 2])
    assert_(van.shape == (1, 5, 6))