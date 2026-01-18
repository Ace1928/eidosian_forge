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
def test_asstr(self):
    ds = self.f.create_dataset('x', (10,), dtype=h5py.string_dtype())
    data = 'fàilte'
    ds[0] = data
    strwrap1 = ds.asstr('ascii')
    with self.assertRaises(UnicodeDecodeError):
        out = strwrap1[0]
    self.assertEqual(ds.asstr('ascii', 'ignore')[0], 'filte')
    self.assertNotEqual(ds.asstr('latin-1')[0], data)
    self.assertEqual(10, len(ds.asstr()))
    np.testing.assert_array_equal(ds.asstr()[:1], np.array([data], dtype=object))
    np.testing.assert_array_equal(np.asarray(ds.asstr())[:1], np.array([data], dtype=object))