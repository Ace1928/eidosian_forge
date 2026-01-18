import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
def test_get_dimension(self):
    with self.assertRaises(IndexError):
        self.f['data'].dims[3]