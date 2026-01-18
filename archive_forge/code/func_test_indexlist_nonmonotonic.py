import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_indexlist_nonmonotonic(self):
    """ we require index list values to be strictly increasing """
    with self.assertRaises(TypeError):
        self.dset[[1, 3, 2]]