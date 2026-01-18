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
def test_dtype_conflict(self):
    """ require_dataset with dtype conflict (strict mode) yields TypeError
        """
    dset = self.f.create_dataset('foo', (10, 3), 'f')
    with self.assertRaises(TypeError):
        self.f.require_dataset('foo', (10, 3), 'S10')