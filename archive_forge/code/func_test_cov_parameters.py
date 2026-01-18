import os
import numpy as np
from numpy.testing import (
def test_cov_parameters(self):
    x = np.random.random((3, 3))
    y = x.copy()
    np.cov(x, rowvar=True)
    np.cov(y, rowvar=False)
    assert_array_equal(x, y)