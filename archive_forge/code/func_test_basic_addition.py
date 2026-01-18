import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
def test_basic_addition(self):
    a = self._get_array(2.0)
    b = self._get_array(4.0)
    res = a + b
    assert res.dtype == np.result_type(a.dtype, b.dtype)
    expected_view = a.astype(res.dtype).view(np.float64) + b.astype(res.dtype).view(np.float64)
    assert_array_equal(res.view(np.float64), expected_view)