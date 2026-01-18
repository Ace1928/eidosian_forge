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
def assert_string_type(self, ds, cset, variable=True):
    tid = ds.id.get_type()
    self.assertEqual(type(tid), h5py.h5t.TypeStringID)
    self.assertEqual(tid.get_cset(), cset)
    if variable:
        assert tid.is_variable_str()