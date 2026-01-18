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
@ut.skipUnless(h5py.version.hdf5_version_tuple >= (1, 10, 5), 'chunk info requires  HDF5 >= 1.10.5')
def test_get_chunk_details():
    from io import BytesIO
    buf = BytesIO()
    with h5py.File(buf, 'w') as fout:
        fout.create_dataset('test', shape=(100, 100), chunks=(10, 10), dtype='i4')
        fout['test'][:] = 1
    buf.seek(0)
    with h5py.File(buf, 'r') as fin:
        ds = fin['test'].id
        assert ds.get_num_chunks() == 100
        for j in range(100):
            offset = tuple(np.array(np.unravel_index(j, (10, 10))) * 10)
            si = ds.get_chunk_info(j)
            assert si.chunk_offset == offset
            assert si.filter_mask == 0
            assert si.byte_offset is not None
            assert si.size > 0
        si = ds.get_chunk_info_by_coord((0, 0))
        assert si.chunk_offset == (0, 0)
        assert si.filter_mask == 0
        assert si.byte_offset is not None
        assert si.size > 0