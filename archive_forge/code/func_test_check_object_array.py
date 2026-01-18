import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_check_object_array(self):
    arr = np.empty(1, dtype=object)
    obj_a = object()
    arr[0] = obj_a
    obj_b = object()
    obj_c = object()
    arr = np.pad(arr, pad_width=1, mode='constant', constant_values=(obj_b, obj_c))
    expected = np.empty((3,), dtype=object)
    expected[0] = obj_b
    expected[1] = obj_a
    expected[2] = obj_c
    assert_array_equal(arr, expected)