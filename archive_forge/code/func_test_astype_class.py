import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
def test_astype_class(self):
    arr = np.array([1.0, 2.0, 3.0], dtype=object)
    res = arr.astype(SF)
    expected = arr.astype(SF(1.0))
    assert_array_equal(res.view(np.float64), expected.view(np.float64))