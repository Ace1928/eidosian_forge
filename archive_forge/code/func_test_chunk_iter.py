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
@ut.skipUnless(h5py.version.hdf5_version_tuple >= (1, 12, 3) or (h5py.version.hdf5_version_tuple >= (1, 10, 10) and h5py.version.hdf5_version_tuple < (1, 10, 99)), 'chunk iteration requires  HDF5 1.10.10 and later 1.10, or 1.12.3 and later')
def test_chunk_iter():
    """H5Dchunk_iter() for chunk information"""
    from io import BytesIO
    buf = BytesIO()
    with h5py.File(buf, 'w') as f:
        f.create_dataset('test', shape=(100, 100), chunks=(10, 10), dtype='i4')
        f['test'][:] = 1
    buf.seek(0)
    with h5py.File(buf, 'r') as f:
        dsid = f['test'].id
        num_chunks = dsid.get_num_chunks()
        assert num_chunks == 100
        ci = {}
        for j in range(num_chunks):
            si = dsid.get_chunk_info(j)
            ci[si.chunk_offset] = si

        def callback(chunk_info):
            known = ci[chunk_info.chunk_offset]
            assert chunk_info.chunk_offset == known.chunk_offset
            assert chunk_info.filter_mask == known.filter_mask
            assert chunk_info.byte_offset == known.byte_offset
            assert chunk_info.size == known.size
        dsid.chunk_iter(callback)