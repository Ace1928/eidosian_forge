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
def test_check_page_buf_size(self):
    """Verify set page buffer size, and minimum meta and raw eviction criteria."""
    fname = self.mktemp()
    pbs = 16 * 1024
    mm = 19
    mr = 67
    with File(fname, mode='w', fs_strategy='page', page_buf_size=pbs, min_meta_keep=mm, min_raw_keep=mr) as f:
        fapl = f.id.get_access_plist()
        self.assertEqual(fapl.get_page_buffer_size(), (pbs, mm, mr))