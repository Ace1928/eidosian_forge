from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_b8_bool_compound(self):
    arr1 = np.array([(False,), (True,)], dtype=np.dtype([('x', '?')]))
    self._test_b8(arr1, expected_default_cast_dtype=np.dtype([('x', 'u1')]))
    self._test_b8(arr1, expected_default_cast_dtype=np.dtype([('x', 'u1')]), cast_dtype=np.dtype([('x', 'u1')]))