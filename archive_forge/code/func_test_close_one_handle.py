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
def test_close_one_handle(self):
    fname = self.mktemp()
    with File(fname, 'w') as f:
        f.create_group('foo')
    f1 = File(fname)
    f2 = File(fname)
    g1 = f1['foo']
    g2 = f2['foo']
    assert g1.id.valid
    assert g2.id.valid
    f1.close()
    assert not g1.id.valid
    assert f2.id.valid
    assert g2.id.valid
    f2.close()
    assert not f2.id.valid
    assert not g2.id.valid