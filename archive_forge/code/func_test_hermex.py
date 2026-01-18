from functools import reduce
import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermex(self):
    assert_equal(herme.hermex, [0, 1])