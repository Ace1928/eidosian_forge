import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_block_more_than_stride_error(self):
    with self.assertRaises(ValueError):
        MultiBlockSlice(block=3)
    with self.assertRaises(ValueError):
        MultiBlockSlice(stride=2, block=3)