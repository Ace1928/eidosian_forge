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
def test_create_intermediate(self):
    """ Intermediate groups can be created automatically """
    grp = self.f.create_group('foo/bar/baz')
    self.assertEqual(grp.name, '/foo/bar/baz')
    grp2 = self.f.create_group(b'boo/bar/baz')
    self.assertEqual(grp2.name, '/boo/bar/baz')