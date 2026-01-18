from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_b8_bool_compound_mixed_types(self):
    arr1 = np.array([(True, 0.5), (False, 0.2)], dtype=np.dtype([('x', '?'), ('y', '<f8')]))
    self._test_b8(arr1, expected_default_cast_dtype=np.dtype([('x', 'u1'), ('y', '<f8')]))
    self._test_b8(arr1, expected_default_cast_dtype=np.dtype([('x', 'u1'), ('y', '<f8')]), cast_dtype=np.dtype([('x', 'u1'), ('y', '<f8')]))