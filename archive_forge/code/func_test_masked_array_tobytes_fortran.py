import numpy as np
from numpy.testing import (
def test_masked_array_tobytes_fortran(self):
    ma = np.ma.arange(4).reshape((2, 2))
    assert_array_equal(ma.tobytes(order='F'), ma.T.tobytes())