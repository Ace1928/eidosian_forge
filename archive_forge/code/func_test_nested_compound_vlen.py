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
def test_nested_compound_vlen(self):
    dt_inner = np.dtype([('a', h5py.vlen_dtype(np.int32)), ('b', h5py.vlen_dtype(np.int32))])
    dt = np.dtype([('f1', h5py.vlen_dtype(dt_inner)), ('f2', np.int64)])
    inner1 = (np.array(range(1, 3), dtype=np.int32), np.array(range(6, 9), dtype=np.int32))
    inner2 = (np.array(range(10, 14), dtype=np.int32), np.array(range(16, 21), dtype=np.int32))
    data = np.array([(np.array([inner1, inner2], dtype=dt_inner), 2), (np.array([inner1], dtype=dt_inner), 3)], dtype=dt)
    self.f['ds'] = data
    out = self.f['ds']
    self.assertArrayEqual(out, data, check_alignment=False)