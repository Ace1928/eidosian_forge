import numpy as np
from .common import TestCase, ut
import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace
def test_modify(self):
    with self.assertRaises(IOError):
        self.f.attrs.modify('x', 1)