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
def test_compression_number_invalid(self):
    """ Create with invalid compression numbers  """
    with self.assertRaises(ValueError) as e:
        self.f.create_dataset('foo', (20, 30), compression=-999)
    self.assertIn('Invalid filter', str(e.exception))
    with self.assertRaises(ValueError) as e:
        self.f.create_dataset('foo', (20, 30), compression=100)
    self.assertIn('Unknown compression', str(e.exception))
    original_compression_vals = h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS
    try:
        h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS = tuple()
        with self.assertRaises(IndexError):
            self.f.create_dataset('foo', (20, 30), compression=h5py.h5z.FILTER_DEFLATE)
    finally:
        h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS = original_compression_vals