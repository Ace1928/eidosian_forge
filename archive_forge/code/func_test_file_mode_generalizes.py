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
def test_file_mode_generalizes(self):
    fname = self.mktemp()
    fid = File(fname, 'w', libver='latest')
    g = fid.create_group('foo')
    assert fid.mode == g.file.mode == 'r+'
    fid.swmr_mode = True
    assert fid.mode == g.file.mode == 'r+'
    fid.close()