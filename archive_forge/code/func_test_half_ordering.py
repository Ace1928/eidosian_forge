import platform
import pytest
import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM
def test_half_ordering(self):
    """Make sure comparisons are working right"""
    a = self.nonan_f16[::-1].copy()
    b = np.array(a, dtype=float32)
    a.sort()
    b.sort()
    assert_equal(a, b)
    assert_((a[:-1] <= a[1:]).all())
    assert_(not (a[:-1] > a[1:]).any())
    assert_((a[1:] >= a[:-1]).all())
    assert_(not (a[1:] < a[:-1]).any())
    assert_equal(np.nonzero(a[:-1] < a[1:])[0].size, a.size - 2)
    assert_equal(np.nonzero(a[1:] > a[:-1])[0].size, a.size - 2)