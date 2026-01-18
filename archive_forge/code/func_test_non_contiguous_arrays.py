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
def test_non_contiguous_arrays(self):
    """Test that non-contiguous arrays are stored correctly"""
    self.f.create_dataset('nc', (10,), dtype=h5py.vlen_dtype('bool'))
    x = np.array([True, False, True, True, False, False, False])
    self.f['nc'][0] = x[::2]
    assert all(self.f['nc'][0] == x[::2]), f'{self.f['nc'][0]} != {x[::2]}'
    self.f.create_dataset('nc2', (10,), dtype=h5py.vlen_dtype('int8'))
    y = np.array([2, 4, 1, 5, -1, 3, 7])
    self.f['nc2'][0] = y[::2]
    assert all(self.f['nc2'][0] == y[::2]), f'{self.f['nc2'][0]} != {y[::2]}'