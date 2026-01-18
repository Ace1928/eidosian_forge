import pytest
import os
import stat
import pickle
import tempfile
import subprocess
import sys
from .common import ut, TestCase, UNICODE_FILENAMES, closed_tempfile
from h5py._hl.files import direct_vfd
from h5py import File
import h5py
from .. import h5
import pathlib
import sys
import h5py
def test_create_exclusive(self):
    """ Mode 'w-' opens file in exclusive mode """
    fname = self.mktemp()
    fid = File(fname, 'w-')
    self.assertTrue(fid)
    fid.close()
    with self.assertRaises(FileExistsError):
        File(fname, 'w-')