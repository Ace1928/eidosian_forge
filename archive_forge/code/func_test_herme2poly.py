from functools import reduce
import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_herme2poly(self):
    for i in range(10):
        assert_almost_equal(herme.herme2poly([0] * i + [1]), Helist[i])