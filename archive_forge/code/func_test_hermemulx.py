from functools import reduce
import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermemulx(self):
    assert_equal(herme.hermemulx([0]), [0])
    assert_equal(herme.hermemulx([1]), [0, 1])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [i, 0, 1]
        assert_equal(herme.hermemulx(ser), tgt)