import numpy as np
import os
import os.path
import sys
from tempfile import mkdtemp
from collections.abc import MutableMapping
from .common import ut, TestCase
import h5py
from h5py import File, Group, SoftLink, HardLink, ExternalLink
from h5py import Dataset, Datatype
from h5py import h5t
from h5py._hl.compat import filename_encode
def test_copy_dataset(self):
    self.f1['foo'] = [1, 2, 3]
    foo = self.f1['foo']
    grp = self.f1.create_group('grp')
    self.f1.copy(foo, 'bar')
    self.assertArrayEqual(self.f1['bar'], np.array([1, 2, 3]))
    self.f1.copy('foo', 'baz')
    self.assertArrayEqual(self.f1['baz'], np.array([1, 2, 3]))
    self.f1.copy(foo, grp)
    self.assertArrayEqual(self.f1['/grp/foo'], np.array([1, 2, 3]))
    self.f1.copy('foo', self.f2)
    self.assertArrayEqual(self.f2['foo'], np.array([1, 2, 3]))
    self.f2.copy(self.f1['foo'], self.f2, 'bar')
    self.assertArrayEqual(self.f2['bar'], np.array([1, 2, 3]))