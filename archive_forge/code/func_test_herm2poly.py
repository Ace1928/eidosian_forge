from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_herm2poly(self):
    for i in range(10):
        assert_almost_equal(herm.herm2poly([0] * i + [1]), Hlist[i])