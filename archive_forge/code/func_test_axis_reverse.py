import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises as assert_raises
from scipy.signal._arraytools import (axis_slice, axis_reverse,
def test_axis_reverse(self):
    a = np.arange(12).reshape(3, 4)
    r = axis_reverse(a, axis=0)
    assert_array_equal(r, a[::-1, :])
    r = axis_reverse(a, axis=1)
    assert_array_equal(r, a[:, ::-1])