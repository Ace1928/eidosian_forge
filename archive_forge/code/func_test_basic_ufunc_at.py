import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
def test_basic_ufunc_at(self):
    float_a = np.array([1.0, 2.0, 3.0])
    b = self._get_array(2.0)
    float_b = b.view(np.float64).copy()
    np.multiply.at(float_b, [1, 1, 1], float_a)
    np.multiply.at(b, [1, 1, 1], float_a)
    assert_array_equal(b.view(np.float64), float_b)