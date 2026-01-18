import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_indexlist_monotonic_negative(self):
    self.assertNumpyBehavior(self.dset, self.data, np.s_[[0, 2, -2]])
    with self.assertRaises(TypeError):
        self.dset[[-2, -3]]