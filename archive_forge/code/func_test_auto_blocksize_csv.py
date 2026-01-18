from __future__ import annotations
import gzip
import os
import warnings
from io import BytesIO, StringIO
from unittest import mock
import pytest
import fsspec
from fsspec.compression import compr
from packaging.version import Version
from tlz import partition_all, valmap
import dask
from dask.base import compute_as_if_collection
from dask.bytes.core import read_bytes
from dask.bytes.utils import compress
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.io.csv import (
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import (
from dask.layers import DataFrameIOLayer
from dask.utils import filetext, filetexts, tmpdir, tmpfile
from dask.utils_test import hlg_layer
def test_auto_blocksize_csv(monkeypatch):
    psutil = pytest.importorskip('psutil')
    total_memory = psutil.virtual_memory().total
    cpu_count = psutil.cpu_count()
    mock_read_bytes = mock.Mock(wraps=read_bytes)
    monkeypatch.setattr(dask.dataframe.io.csv, 'read_bytes', mock_read_bytes)
    expected_block_size = auto_blocksize(total_memory, cpu_count)
    with filetexts(csv_files, mode='b'):
        dd.read_csv('2014-01-01.csv')
        assert mock_read_bytes.called
        assert mock_read_bytes.call_args[1]['blocksize'] == expected_block_size