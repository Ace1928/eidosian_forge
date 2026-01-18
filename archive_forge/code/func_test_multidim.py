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
def test_multidim(self):
    dt = h5py.vlen_dtype(int)
    ds = self.f.create_dataset('vlen', (2, 2), dtype=dt)
    ds[0, 0] = np.arange(1)
    ds[:, :] = np.array([[np.arange(3), np.arange(2)], [np.arange(1), np.arange(2)]], dtype=object)
    ds[:, :] = np.array([[np.arange(2), np.arange(2)], [np.arange(2), np.arange(2)]])