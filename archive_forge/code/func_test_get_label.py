import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
def test_get_label(self):
    self.assertEqual(self.f['data'].dims[2].label, 'x')
    self.assertEqual(self.f['data'].dims[1].label, '')
    self.assertEqual(self.f['data'].dims[0].label, 'z')
    self.assertEqual(self.f['data2'].dims[2].label, '')
    self.assertEqual(self.f['data2'].dims[1].label, '')
    self.assertEqual(self.f['data2'].dims[0].label, '')