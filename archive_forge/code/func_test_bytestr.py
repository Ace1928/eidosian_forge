import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_bytestr(self):
    """ Indexing a byte string dataset returns a real python byte string
        """
    dset = self.f.create_dataset('x', (1,), dtype=h5py.string_dtype(encoding='ascii'))
    dset[0] = b'Hello there!'
    self.assertEqual(type(dset[0]), bytes)