from functools import reduce
import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermeroots(self):
    assert_almost_equal(herme.hermeroots([1]), [])
    assert_almost_equal(herme.hermeroots([1, 1]), [-1])
    for i in range(2, 5):
        tgt = np.linspace(-1, 1, i)
        res = herme.hermeroots(herme.hermefromroots(tgt))
        assert_almost_equal(trim(res), trim(tgt))