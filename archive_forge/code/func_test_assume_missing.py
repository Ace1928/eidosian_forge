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
def test_assume_missing():
    text = 'numbers,names,more_numbers,integers\n'
    for _ in range(1000):
        text += '1,foo,2,3\n'
    text += '1.5,bar,2.5,3\n'
    with filetext(text) as fn:
        sol = pd.read_csv(fn)
        res = dd.read_csv(fn, sample=50, assume_missing=True)
        assert_eq(res, sol.astype({'integers': float}))
        res = dd.read_csv(fn, sample=50, assume_missing=True, dtype={'integers': 'int64'})
        assert_eq(res, sol)
        res = dd.read_csv(fn, sample=50, assume_missing=True, dtype=None)
        assert_eq(res, sol.astype({'integers': float}))
    text = 'numbers,integers\n'
    for _ in range(1000):
        text += '1,2\n'
    text += '1.5,2\n'
    with filetext(text) as fn:
        sol = pd.read_csv(fn)
        df = dd.read_csv(fn, sample=30, dtype='int64', assume_missing=True)
        assert df.numbers.dtype == 'int64'