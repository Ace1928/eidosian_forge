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
def test_create_blocksize(self):
    """ User blocks created with w, w-, x and properties work correctly """
    f = File(self.mktemp(), 'w-', userblock_size=512)
    try:
        self.assertEqual(f.userblock_size, 512)
    finally:
        f.close()
    f = File(self.mktemp(), 'x', userblock_size=512)
    try:
        self.assertEqual(f.userblock_size, 512)
    finally:
        f.close()
    f = File(self.mktemp(), 'w', userblock_size=512)
    try:
        self.assertEqual(f.userblock_size, 512)
    finally:
        f.close()
    with self.assertRaises(ValueError):
        File(self.mktemp(), 'w', userblock_size='non')