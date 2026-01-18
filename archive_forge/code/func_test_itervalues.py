import numpy as np
from .common import TestCase, ut
import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace
def test_itervalues(self):
    values = list(self.f.attrs.values())
    self.assertEqual([self.empty_obj], values)