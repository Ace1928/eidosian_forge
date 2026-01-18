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
def test_named(self):
    """ Named type object works and links the dataset to type """
    self.f['type'] = np.dtype('f8')
    dset = self.f.create_dataset('x', (100,), dtype=self.f['type'])
    self.assertEqual(dset.dtype, np.dtype('f8'))
    self.assertEqual(dset.id.get_type(), self.f['type'].id)
    self.assertTrue(dset.id.get_type().committed())