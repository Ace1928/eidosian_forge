from functools import reduce
import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermetrim(self):
    coef = [2, -1, 1, 0]
    assert_raises(ValueError, herme.hermetrim, coef, -1)
    assert_equal(herme.hermetrim(coef), coef[:-1])
    assert_equal(herme.hermetrim(coef, 1), coef[:-3])
    assert_equal(herme.hermetrim(coef, 2), [0])