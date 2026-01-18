import numpy as np
from .common import TestCase, ut
import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace
def test_compound(self):
    """ Compound scalars are read as numpy.void """
    dt = np.dtype([('a', 'i'), ('b', 'f')])
    data = np.array((1, 4.2), dtype=dt)
    self.f.attrs['x'] = data
    out = self.f.attrs['x']
    self.assertIsInstance(out, np.void)
    self.assertEqual(out, data)
    self.assertEqual(out['b'], data['b'])