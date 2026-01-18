from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermroots(self):
    assert_almost_equal(herm.hermroots([1]), [])
    assert_almost_equal(herm.hermroots([1, 1]), [-0.5])
    for i in range(2, 5):
        tgt = np.linspace(-1, 1, i)
        res = herm.hermroots(herm.hermfromroots(tgt))
        assert_almost_equal(trim(res), trim(tgt))