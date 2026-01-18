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
def test_fields(self):
    dt = np.dtype([('x', np.float64), ('y', np.float64), ('z', np.float64)])
    testdata = np.ndarray((16,), dtype=dt)
    for key in dt.fields:
        testdata[key] = np.random.random((16,)) * 100
    self.f['test'] = testdata
    np.testing.assert_array_equal(self.f['test'].fields(['x', 'y'])[:], testdata[['x', 'y']])
    np.testing.assert_array_equal(self.f['test'].fields('x')[:], testdata['x'])
    np.testing.assert_array_equal(np.asarray(self.f['test'].fields(['x', 'y'])), testdata[['x', 'y']])
    dt_int = np.dtype([('x', np.int32)])
    np.testing.assert_array_equal(np.asarray(self.f['test'].fields(['x']), dtype=dt_int), testdata[['x']].astype(dt_int))
    assert len(self.f['test'].fields('x')) == 16