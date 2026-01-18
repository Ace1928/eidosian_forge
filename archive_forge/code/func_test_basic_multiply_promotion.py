import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
def test_basic_multiply_promotion(self):
    float_a = np.array([1.0, 2.0, 3.0])
    b = self._get_array(2.0)
    res1 = float_a * b
    res2 = b * float_a
    assert res1.dtype == res2.dtype == b.dtype
    expected_view = float_a * b.view(np.float64)
    assert_array_equal(res1.view(np.float64), expected_view)
    assert_array_equal(res2.view(np.float64), expected_view)
    np.multiply(b, float_a, out=res2)
    with pytest.raises(TypeError):
        np.multiply(b, float_a, out=np.arange(3))