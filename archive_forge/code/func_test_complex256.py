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
@ut.skipIf(not hasattr(np, 'complex256'), 'No support for complex256')
def test_complex256(self):
    """ Confirm that the default dtype is float """
    dset = self.f.create_dataset('foo', (63,), dtype=np.dtype('complex256'))
    self.assertEqual(dset.dtype, np.dtype('complex256'))