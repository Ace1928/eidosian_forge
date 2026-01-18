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
def test_mode_attr(self):
    """ Mode equivalent can be retrieved via property """
    fname = self.mktemp()
    with File(fname, 'w') as f:
        self.assertEqual(f.mode, 'r+')
    with File(fname, 'r') as f:
        self.assertEqual(f.mode, 'r')