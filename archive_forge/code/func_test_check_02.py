import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_check_02(self):
    a = np.pad([1, 2, 3], 4, 'wrap')
    b = np.array([3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
    assert_array_equal(a, b)