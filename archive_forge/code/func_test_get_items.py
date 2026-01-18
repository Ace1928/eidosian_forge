import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
def test_get_items(self):
    self.assertEqual(self.f['data'].dims[2].items(), [('', self.f['x1']), ('x2 name', self.f['x2'])])