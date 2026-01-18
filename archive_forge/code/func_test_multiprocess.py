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
def test_multiprocess(self, tmp_path):
    """Test file locking option from different concurrent processes"""
    fname = tmp_path / 'test.h5'

    def open_in_subprocess(filename, mode, locking):
        """Open HDF5 file in a subprocess and return True on success"""
        h5py_import_dir = str(pathlib.Path(h5py.__file__).parent.parent)
        process = subprocess.run([sys.executable, '-c', f'\nimport sys\nsys.path.insert(0, {h5py_import_dir!r})\nimport h5py\nf = h5py.File({str(filename)!r}, mode={mode!r}, locking={locking})\n                    '], capture_output=True)
        return process.returncode == 0 and (not process.stderr)
    with h5py.File(fname, mode='w', locking=True) as f:
        f['data'] = 1
    with h5py.File(fname, mode='r', locking=False) as f:
        assert open_in_subprocess(fname, mode='w', locking=True)