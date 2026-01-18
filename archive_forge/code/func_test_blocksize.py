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
def test_blocksize(self):
    """ Core driver supports variable block size """
    fname = self.mktemp()
    fid = File(fname, 'w', driver='core', block_size=1024, backing_store=False)
    self.assertTrue(fid)
    fid.close()