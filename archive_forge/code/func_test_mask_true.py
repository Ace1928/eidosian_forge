import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_mask_true(self):
    self.assertNumpyBehavior(self.dset, self.data, np.s_[self.data > -100], skip_fast_reader=True)