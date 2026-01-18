import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
@pytest.mark.parametrize('kind', [None, 'sort', 'table'])
def test_in1d_invert(self, kind):
    """Test in1d's invert parameter"""
    for mult in (1, 10):
        a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5])
        b = [2, 3, 4] * mult
        assert_array_equal(np.invert(in1d(a, b, kind=kind)), in1d(a, b, invert=True, kind=kind))
    if kind in {None, 'sort'}:
        for mult in (1, 10):
            a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5], dtype=np.float32)
            b = [2, 3, 4] * mult
            b = np.array(b, dtype=np.float32)
            assert_array_equal(np.invert(in1d(a, b, kind=kind)), in1d(a, b, invert=True, kind=kind))