import numpy as np
from collections.abc import MutableMapping
from .common import TestCase, ut
import h5py
from h5py import File
from h5py import h5a,  h5t
from h5py import AttributeManager
def test_modify_2(self):
    """ Attributes are modified by modify() method """
    self.f.attrs.modify('a', 3)
    self.assertEqual(list(self.f.attrs.keys()), ['a'])
    self.assertEqual(self.f.attrs['a'], 3)
    self.f.attrs.modify('a', 4)
    self.assertEqual(list(self.f.attrs.keys()), ['a'])
    self.assertEqual(self.f.attrs['a'], 4)
    self.f.attrs.modify('b', 5)
    self.assertEqual(list(self.f.attrs.keys()), ['a', 'b'])
    self.assertEqual(self.f.attrs['a'], 4)
    self.assertEqual(self.f.attrs['b'], 5)
    new_value = np.arange(5)
    with self.assertRaises(TypeError):
        self.f.attrs.modify('b', new_value)