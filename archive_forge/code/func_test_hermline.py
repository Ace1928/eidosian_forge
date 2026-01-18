from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermline(self):
    assert_equal(herm.hermline(3, 4), [3, 2])