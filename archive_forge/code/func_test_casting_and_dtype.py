import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def test_casting_and_dtype(self):
    a = np.array([1, 2, 3])
    b = np.array([2.5, 3.5, 4.5])
    res = np.vstack((a, b), casting='unsafe', dtype=np.int64)
    expected_res = np.array([[1, 2, 3], [2, 3, 4]])
    assert_array_equal(res, expected_res)