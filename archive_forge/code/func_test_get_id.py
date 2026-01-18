import numpy as np
from collections.abc import MutableMapping
from .common import TestCase, ut
import h5py
from h5py import File
from h5py import h5a,  h5t
from h5py import AttributeManager
def test_get_id(self):
    self.f.attrs['a'] = 4.0
    aid = self.f.attrs.get_id('a')
    assert isinstance(aid, h5a.AttrID)
    with self.assertRaises(KeyError):
        self.f.attrs.get_id('b')