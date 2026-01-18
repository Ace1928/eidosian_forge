from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def assertVlenArrayEqual(self, dset, arr, message=None, precision=None):
    assert dset.shape == arr.shape, 'Shape mismatch (%s vs %s)%s' % (dset.shape, arr.shape, message)
    for i, d, a in zip(count(), dset, arr):
        self.assertArrayEqual(d, a, message, precision)