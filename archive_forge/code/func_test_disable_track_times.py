import pathlib
import os
import sys
import numpy as np
import platform
import pytest
import warnings
from .common import ut, TestCase
from .data_files import get_data_file_path
from h5py import File, Group, Dataset
from h5py._hl.base import is_empty_dataspace, product
from h5py import h5f, h5t
from h5py.h5py_warnings import H5pyDeprecationWarning
from h5py import version
import h5py
import h5py._hl.selections as sel
def test_disable_track_times(self):
    """ check that when track_times=False, the time stamp=0 (Jan 1, 1970) """
    ds = self.f.create_dataset('foo', (4,), track_times=False)
    ds_mtime = h5py.h5g.get_objinfo(ds._id).mtime
    self.assertEqual(0, ds_mtime)