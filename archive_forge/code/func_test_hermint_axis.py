from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermint_axis(self):
    c2d = np.random.random((3, 4))
    tgt = np.vstack([herm.hermint(c) for c in c2d.T]).T
    res = herm.hermint(c2d, axis=0)
    assert_almost_equal(res, tgt)
    tgt = np.vstack([herm.hermint(c) for c in c2d])
    res = herm.hermint(c2d, axis=1)
    assert_almost_equal(res, tgt)
    tgt = np.vstack([herm.hermint(c, k=3) for c in c2d])
    res = herm.hermint(c2d, k=3, axis=1)
    assert_almost_equal(res, tgt)