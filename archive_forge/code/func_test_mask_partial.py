import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_mask_partial(self):
    self.assertNumpyBehavior(self.dset, self.data, np.s_[self.data > 5], skip_fast_reader=True)