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
def test_dset_chunk_cache():
    """Chunk cache configuration for individual datasets."""
    from io import BytesIO
    buf = BytesIO()
    with h5py.File(buf, 'w') as fout:
        ds = fout.create_dataset('x', shape=(10, 20), chunks=(5, 4), dtype='i4', rdcc_nbytes=2 * 1024 * 1024, rdcc_w0=0.2, rdcc_nslots=997)
        ds_chunk_cache = ds.id.get_access_plist().get_chunk_cache()
        assert fout.id.get_access_plist().get_cache()[1:] != ds_chunk_cache
        assert ds_chunk_cache == (997, 2 * 1024 * 1024, 0.2)
    buf.seek(0)
    with h5py.File(buf, 'r') as fin:
        ds = fin.require_dataset('x', shape=(10, 20), dtype='i4', rdcc_nbytes=3 * 1024 * 1024, rdcc_w0=0.67, rdcc_nslots=709)
        ds_chunk_cache = ds.id.get_access_plist().get_chunk_cache()
        assert fin.id.get_access_plist().get_cache()[1:] != ds_chunk_cache
        assert ds_chunk_cache == (709, 3 * 1024 * 1024, 0.67)