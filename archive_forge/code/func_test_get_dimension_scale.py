import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
def test_get_dimension_scale(self):
    self.assertEqual(self.f['data'].dims[2][0], self.f['x1'])
    with self.assertRaises(RuntimeError):
        (self.f['data2'].dims[2][0], self.f['x2'])
    self.assertEqual(self.f['data'].dims[2][''], self.f['x1'])
    self.assertEqual(self.f['data'].dims[2]['x2 name'], self.f['x2'])