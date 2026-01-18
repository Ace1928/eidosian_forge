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
def test_int_with_minbits(self):
    """ Scaleoffset filter works for integer data with specified precision """
    nbits = 12
    shape = (100, 300)
    testdata = np.random.randint(0, 2 ** nbits, size=shape)
    dset = self.f.create_dataset('foo', shape, dtype=int, scaleoffset=nbits)
    self.assertTrue(dset.scaleoffset == 12)
    dset[...] = testdata
    filename = self.f.filename
    self.f.close()
    self.f = h5py.File(filename, 'r')
    readdata = self.f['foo'][...]
    self.assertArrayEqual(readdata, testdata)