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
def test_get_link(self):
    """ Get link values """
    sl = SoftLink('/mongoose')
    el = ExternalLink('somewhere.hdf5', 'mongoose')
    self.f.create_group('hard')
    self.f['soft'] = sl
    self.f['external'] = el
    out_hl = self.f.get('hard', getlink=True)
    out_sl = self.f.get('soft', getlink=True)
    out_el = self.f.get('external', getlink=True)
    self.assertIsInstance(out_hl, HardLink)
    self.assertIsInstance(out_sl, SoftLink)
    self.assertEqual(out_sl._path, sl._path)
    self.assertIsInstance(out_el, ExternalLink)
    self.assertEqual(out_el._path, el._path)
    self.assertEqual(out_el._filename, el._filename)