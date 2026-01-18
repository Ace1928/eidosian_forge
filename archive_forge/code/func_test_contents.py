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
def test_contents(self):
    """ Create and access an external dataset """
    shape = (6, 100)
    testdata = np.random.random(shape)
    ext_file = self.mktemp()
    external = [(ext_file, 0, h5f.UNLIMITED)]
    dset = self.f.create_dataset('foo', shape, dtype=testdata.dtype, external=external, efile_prefix='${ORIGIN}')
    dset[...] = testdata
    assert dset.external is not None
    with open(ext_file, 'rb') as fid:
        contents = fid.read()
    assert contents == testdata.tobytes()
    efile_prefix = pathlib.Path(dset.id.get_access_plist().get_efile_prefix().decode()).as_posix()
    parent = pathlib.Path(self.f.filename).parent.as_posix()
    assert efile_prefix == parent