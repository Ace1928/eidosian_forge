import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_index_outofrange(self):
    with self.assertRaises(IndexError):
        self.dset[100]