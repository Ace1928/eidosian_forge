import numpy as np
from numpy.testing import (
def test_empty_list_on_structured(self):
    ma = np.ma.MaskedArray([(1, 1.0), (2, 2.0), (3, 3.0)], dtype='i4,f4')
    assert_array_equal(ma[[]], ma[:0])