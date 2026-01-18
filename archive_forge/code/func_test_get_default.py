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
def test_get_default(self):
    """ Object is returned, or default if it doesn't exist """
    default = object()
    out = self.f.get('mongoose', default)
    self.assertIs(out, default)
    grp = self.f.create_group('a')
    out = self.f.get(b'a')
    self.assertEqual(out, grp)