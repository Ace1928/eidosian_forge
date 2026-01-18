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
def test_asstr_fixed(self):
    dt = h5py.string_dtype(length=5)
    ds = self.f.create_dataset('x', (10,), dtype=dt)
    data = 'cù'
    ds[0] = np.array(data.encode('utf-8'), dtype=dt)
    self.assertIsInstance(ds[0], np.bytes_)
    out = ds.asstr()[0]
    self.assertIsInstance(out, str)
    self.assertEqual(out, data)
    self.assertEqual(ds.asstr('ascii', 'ignore')[0], 'c')
    self.assertNotEqual(ds.asstr('latin-1')[0], data)
    np.testing.assert_array_equal(ds.asstr()[:1], np.array([data], dtype=object))