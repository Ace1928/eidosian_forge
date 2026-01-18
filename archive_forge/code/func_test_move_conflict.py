import numpy as np
import os
import os.path
import sys
from tempfile import mkdtemp
from collections.abc import MutableMapping
from .common import ut, TestCase
import h5py
from h5py import File, Group, SoftLink, HardLink, ExternalLink
from h5py import Dataset, Datatype
from h5py import h5t
from h5py._hl.compat import filename_encode
def test_move_conflict(self):
    """ Move conflict raises ValueError """
    self.f.create_group('X')
    self.f.create_group('Y')
    with self.assertRaises(ValueError):
        self.f.move('X', 'Y')