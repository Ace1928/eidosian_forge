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
def test_allow_unknown_filter(writable_file):
    fake_filter_id = 256
    ds = writable_file.create_dataset('data', shape=(10, 10), dtype=np.uint8, compression=fake_filter_id, allow_unknown_filter=True)
    assert str(fake_filter_id) in ds._filters