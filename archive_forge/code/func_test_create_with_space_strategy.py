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
def test_create_with_space_strategy(self):
    """ Create file with file space strategy """
    fname = self.mktemp()
    fid = File(fname, 'w', fs_strategy='page', fs_persist=True, fs_threshold=100)
    self.assertTrue(fid)
    with self.assertRaises(ValueError):
        File(fname, 'a', fs_strategy='page')
    with self.assertRaises(ValueError):
        File(self.mktemp(), 'w', fs_strategy='invalid')
    dset = fid.create_dataset('foo', (100,), dtype='uint8')
    dset[...] = 1
    dset = fid.create_dataset('bar', (100,), dtype='uint8')
    dset[...] = 1
    del fid['foo']
    fid.close()
    fid = File(fname, 'a')
    plist = fid.id.get_create_plist()
    fs_strat = plist.get_file_space_strategy()
    assert fs_strat[0] == 1
    assert fs_strat[1] == True
    assert fs_strat[2] == 100
    dset = fid.create_dataset('foo2', (100,), dtype='uint8')
    dset[...] = 1
    fid.close()