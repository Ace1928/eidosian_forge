import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
def test_label_dimensionscale(self):
    self.assertEqual(h5py.h5ds.get_label(self.f['data'].id, 0), b'z')
    self.assertEqual(h5py.h5ds.get_label(self.f['data'].id, 1), b'')
    self.assertEqual(h5py.h5ds.get_label(self.f['data'].id, 2), b'x')