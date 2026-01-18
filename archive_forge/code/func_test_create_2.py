import numpy as np
from collections.abc import MutableMapping
from .common import TestCase, ut
import h5py
from h5py import File
from h5py import h5a,  h5t
from h5py import AttributeManager
def test_create_2(self):
    """ Attribute creation by create() method """
    self.f.attrs.create('a', 4.0)
    self.assertEqual(list(self.f.attrs.keys()), ['a'])
    self.assertEqual(self.f.attrs['a'], 4.0)