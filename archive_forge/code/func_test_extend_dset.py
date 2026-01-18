import numpy as np
import h5py
from .common import ut, TestCase
def test_extend_dset(self):
    """ Extend and flush a SWMR dataset
        """
    self.f.swmr_mode = True
    self.assertTrue(self.f.swmr_mode)
    self.dset.resize(self.data.shape)
    self.dset[:] = self.data
    self.dset.flush()
    self.dset.refresh()
    self.assertArrayEqual(self.dset, self.data)