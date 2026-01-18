import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
def test_complex_bad2(self):
    with np.errstate(divide='ignore', invalid='ignore'):
        v = 1 + 1j
        v += np.array(-1 + 1j) / 0.0
    vals = nan_to_num(v)
    assert_all(np.isfinite(vals))
    assert_equal(type(vals), np.complex_)