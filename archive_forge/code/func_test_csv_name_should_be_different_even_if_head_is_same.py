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
def test_csv_name_should_be_different_even_if_head_is_same(tmpdir):
    import random
    from shutil import copyfile
    old_csv_path = os.path.join(str(tmpdir), 'old.csv')
    new_csv_path = os.path.join(str(tmpdir), 'new_csv')
    with open(old_csv_path, 'w') as f:
        for _ in range(10):
            f.write(f'{random.randrange(1, 10 ** 9):09}, {random.randrange(1, 10 ** 9):09}, {random.randrange(1, 10 ** 9):09}\n')
    copyfile(old_csv_path, new_csv_path)
    with open(new_csv_path, 'a') as f:
        for _ in range(3):
            f.write(f'{random.randrange(1, 10 ** 9):09}, {random.randrange(1, 10 ** 9):09}, {random.randrange(1, 10 ** 9):09}\n')
    new_df = dd.read_csv(new_csv_path, header=None, delimiter=',', dtype=str, blocksize=None)
    old_df = dd.read_csv(old_csv_path, header=None, delimiter=',', dtype=str, blocksize=None)
    assert new_df.dask.keys() != old_df.dask.keys()