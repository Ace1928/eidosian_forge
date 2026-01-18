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
def test_copy_external_links(self):
    filename = self.f1.filename
    self.f1['foo'] = [1, 2, 3]
    self.f2['bar'] = ExternalLink(filename, 'foo')
    self.f1.close()
    self.f1 = None
    self.assertArrayEqual(self.f2['bar'], np.array([1, 2, 3]))
    self.f2.copy('bar', 'baz', expand_external=True)
    os.unlink(filename)
    self.assertArrayEqual(self.f2['baz'], np.array([1, 2, 3]))