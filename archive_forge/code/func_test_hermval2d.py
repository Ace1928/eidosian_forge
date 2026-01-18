from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermval2d(self):
    x1, x2, x3 = self.x
    y1, y2, y3 = self.y
    assert_raises(ValueError, herm.hermval2d, x1, x2[:2], self.c2d)
    tgt = y1 * y2
    res = herm.hermval2d(x1, x2, self.c2d)
    assert_almost_equal(res, tgt)
    z = np.ones((2, 3))
    res = herm.hermval2d(z, z, self.c2d)
    assert_(res.shape == (2, 3))