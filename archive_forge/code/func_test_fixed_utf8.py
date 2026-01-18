from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_fixed_utf8(self):
    dt = h5py.string_dtype(length=10)
    string_info = h5py.check_string_dtype(dt)
    assert string_info.encoding == 'utf-8'
    assert string_info.length == 10
    assert h5py.check_vlen_dtype(dt) is None