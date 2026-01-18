from functools import reduce
import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermeval2d(self):
    x1, x2, x3 = self.x
    y1, y2, y3 = self.y
    assert_raises(ValueError, herme.hermeval2d, x1, x2[:2], self.c2d)
    tgt = y1 * y2
    res = herme.hermeval2d(x1, x2, self.c2d)
    assert_almost_equal(res, tgt)
    z = np.ones((2, 3))
    res = herme.hermeval2d(z, z, self.c2d)
    assert_(res.shape == (2, 3))