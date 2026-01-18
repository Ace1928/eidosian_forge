import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_negative_stop(self):
    """ Negative stop indexes work as they do in NumPy """
    self.assertArrayEqual(self.dset[2:-2], self.arr[2:-2])