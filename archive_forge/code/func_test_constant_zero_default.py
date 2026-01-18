import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_constant_zero_default():
    arr = np.array([1, 1])
    assert_array_equal(np.pad(arr, 2), [0, 0, 1, 1, 0, 0])