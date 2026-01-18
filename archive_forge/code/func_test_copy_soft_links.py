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
def test_copy_soft_links(self):
    self.f1['bar'] = [1, 2, 3]
    foo = self.f1.create_group('foo')
    foo['baz'] = SoftLink('/bar')
    self.f1.copy(foo, 'qux', expand_soft=True)
    self.f2.copy(foo, 'foo', expand_soft=True)
    del self.f1['bar']
    self.assertIsInstance(self.f1['qux'], Group)
    self.assertArrayEqual(self.f1['qux/baz'], np.array([1, 2, 3]))
    self.assertIsInstance(self.f2['/foo'], Group)
    self.assertArrayEqual(self.f2['foo/baz'], np.array([1, 2, 3]))