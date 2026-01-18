from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_b8_bool(self):
    arr1 = np.array([False, True], dtype=bool)
    self._test_b8(arr1, expected_default_cast_dtype=np.uint8)
    self._test_b8(arr1, expected_default_cast_dtype=np.uint8, cast_dtype=np.uint8)