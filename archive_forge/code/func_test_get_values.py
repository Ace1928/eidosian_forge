import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
def test_get_values(self):
    self.assertEqual(self.f['data'].dims[2].values(), [self.f['x1'], self.f['x2']])