import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
def test_TemporaryFile(self):
    fileobj = tempfile.NamedTemporaryFile()
    fname = fileobj.name
    f = h5py.File(fileobj, 'w')
    del fileobj
    f.create_dataset('test', data=list(range(12)))
    self.assertEqual(list(f), ['test'])
    self.assertEqual(list(f['test'][:]), list(range(12)))
    self.assertTrue(os.path.isfile(fname))
    f.close()
    self.assertFalse(os.path.isfile(fname))