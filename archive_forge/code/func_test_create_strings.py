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
def test_create_strings(self):

    def check_vlen_utf8(dset):
        self.check_h5_string(dset, h5t.CSET_UTF8, length=None)
    check_vlen_utf8(self.f.create_dataset('a', data='abc'))
    check_vlen_utf8(self.f.create_dataset('b', data=['abc', 'def']))
    check_vlen_utf8(self.f.create_dataset('c', data=[['abc'], ['def']]))
    check_vlen_utf8(self.f.create_dataset('d', data=np.array(['abc', 'def'], dtype=object)))