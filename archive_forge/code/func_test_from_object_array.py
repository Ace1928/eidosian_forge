import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_from_object_array(self):
    A = np.array([['abc', 2], ['long   ', '0123456789']], dtype='O')
    B = np.char.array(A)
    assert_equal(B.dtype.itemsize, 10)
    assert_array_equal(B, [[b'abc', b'2'], [b'long', b'0123456789']])