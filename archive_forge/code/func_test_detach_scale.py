import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
def test_detach_scale(self):
    self.f['data'].dims[2].detach_scale(self.f['x1'])
    self.assertEqual(len(self.f['data'].dims[2]), 1)
    self.assertEqual(self.f['data'].dims[2][0], self.f['x2'])
    self.f['data'].dims[2].detach_scale(self.f['x2'])
    self.assertEqual(len(self.f['data'].dims[2]), 0)