import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_count_more_than_length_error(self):
    mbslice = MultiBlockSlice(count=11)
    with self.assertRaises(ValueError):
        mbslice.indices(10)