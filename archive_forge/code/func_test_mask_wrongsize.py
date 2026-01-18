import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_mask_wrongsize(self):
    """ we require the boolean mask shape to match exactly """
    with self.assertRaises(TypeError):
        self.dset[np.ones((2,), dtype='bool')]