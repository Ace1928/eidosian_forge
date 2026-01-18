import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_indexlist_single_index_ellipsis(self):
    self.assertNumpyBehavior(self.dset, self.data, np.s_[[0], ...])