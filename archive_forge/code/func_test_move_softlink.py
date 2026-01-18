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
def test_move_softlink(self):
    """ Moving a soft link """
    self.f['soft'] = h5py.SoftLink('relative/path')
    self.f.move('soft', 'new_soft')
    lnk = self.f.get('new_soft', getlink=True)
    self.assertEqual(lnk.path, 'relative/path')