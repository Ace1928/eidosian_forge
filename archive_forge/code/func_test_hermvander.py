from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermvander(self):
    x = np.arange(3)
    v = herm.hermvander(x, 3)
    assert_(v.shape == (3, 4))
    for i in range(4):
        coef = [0] * i + [1]
        assert_almost_equal(v[..., i], herm.hermval(x, coef))
    x = np.array([[1, 2], [3, 4], [5, 6]])
    v = herm.hermvander(x, 3)
    assert_(v.shape == (3, 2, 4))
    for i in range(4):
        coef = [0] * i + [1]
        assert_almost_equal(v[..., i], herm.hermval(x, coef))