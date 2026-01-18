import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
def test_complex_good(self):
    vals = nan_to_num(1 + 1j)
    assert_all(vals == 1 + 1j)
    assert_equal(type(vals), np.complex_)
    vals = nan_to_num(1 + 1j, nan=10, posinf=20, neginf=30)
    assert_all(vals == 1 + 1j)
    assert_equal(type(vals), np.complex_)