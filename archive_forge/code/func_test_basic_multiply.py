import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
def test_basic_multiply(self):
    a = self._get_array(2.0)
    b = self._get_array(4.0)
    res = a * b
    assert res.dtype.get_scaling() == 8.0
    expected_view = a.view(np.float64) * b.view(np.float64)
    assert_array_equal(res.view(np.float64), expected_view)