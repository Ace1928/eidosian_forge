import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
def test_autoclose(self):
    """ File objects close automatically when out of scope, but
        other objects remain open. """
    start_nfiles = nfiles()
    start_ngroups = ngroups()
    fname = self.mktemp()
    f = h5py.File(fname, 'w')
    g = f['/']
    self.assertEqual(nfiles(), start_nfiles + 1)
    self.assertEqual(ngroups(), start_ngroups + 1)
    del f
    self.assertTrue(g)
    self.assertEqual(nfiles(), start_nfiles)
    self.assertEqual(ngroups(), start_ngroups + 1)
    f = g.file
    self.assertTrue(f)
    self.assertEqual(nfiles(), start_nfiles + 1)
    self.assertEqual(ngroups(), start_ngroups + 1)
    del g
    self.assertEqual(nfiles(), start_nfiles + 1)
    self.assertEqual(ngroups(), start_ngroups)
    del f
    self.assertEqual(nfiles(), start_nfiles)
    self.assertEqual(ngroups(), start_ngroups)