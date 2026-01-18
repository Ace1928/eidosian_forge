import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
def test_complex1(self):
    with np.errstate(divide='ignore', invalid='ignore'):
        assert_all(np.isfinite(np.array(1 + 1j) / 0.0) == 0)