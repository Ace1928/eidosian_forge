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
@pytest.mark.parametrize('dt,expected', [(int, 0), (np.int32, 0), (np.int64, 0), (float, 0.0), (np.float32, 0.0), (np.float64, 0.0), (h5py.string_dtype(encoding='utf-8', length=5), b''), (h5py.string_dtype(encoding='ascii', length=5), b''), (h5py.string_dtype(encoding='utf-8'), b''), (h5py.string_dtype(encoding='ascii'), b''), (h5py.string_dtype(), b'')])
def test_get_unset_fill_value(dt, expected, writable_file):
    dset = writable_file.create_dataset('foo', (10,), dtype=dt)
    assert dset.fillvalue == expected