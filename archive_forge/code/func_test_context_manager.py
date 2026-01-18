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
def test_context_manager(self):
    """ File objects can be used in with statements """
    with File(self.mktemp(), 'w') as fid:
        self.assertTrue(fid)
    self.assertTrue(not fid)