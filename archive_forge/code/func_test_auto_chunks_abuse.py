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
def test_auto_chunks_abuse(self):
    """ Auto-chunking with pathologically large element sizes """
    dset = self.f.create_dataset('foo', shape=(3,), dtype='S100000000', chunks=True)
    self.assertEqual(dset.chunks, (1,))