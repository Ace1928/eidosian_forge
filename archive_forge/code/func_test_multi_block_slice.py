import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_multi_block_slice(self):
    """ MultiBlockSlice -> ValueError """
    with self.assertRaises(ValueError):
        self.dset[h5py.MultiBlockSlice()]