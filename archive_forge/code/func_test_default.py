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
def test_default(self):
    """ Opening with no libver arg """
    f = File(self.mktemp(), 'w')
    self.assertEqual(f.libver, ('earliest', self.latest))
    f.close()