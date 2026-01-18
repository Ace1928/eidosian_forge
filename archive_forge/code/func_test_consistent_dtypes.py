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
def test_consistent_dtypes():
    text = normalize_text('\n    name,amount\n    Alice,100.5\n    Bob,-200.5\n    Charlie,300\n    Dennis,400\n    Edith,-500\n    Frank,600\n    ')
    with filetext(text) as fn:
        df = dd.read_csv(fn, blocksize=30)
        assert df.amount.compute().dtype == float