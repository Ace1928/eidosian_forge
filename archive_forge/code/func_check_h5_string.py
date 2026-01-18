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
def check_h5_string(self, dset, cset, length):
    tid = dset.id.get_type()
    assert isinstance(tid, h5t.TypeStringID)
    assert tid.get_cset() == cset
    if length is None:
        assert tid.is_variable_str()
    else:
        assert not tid.is_variable_str()
        assert tid.get_size() == length