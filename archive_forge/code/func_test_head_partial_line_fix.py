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
def test_head_partial_line_fix():
    files = {'.overflow1.csv': "a,b\n0,'abcdefghijklmnopqrstuvwxyz'\n1,'abcdefghijklmnopqrstuvwxyz'", '.overflow2.csv': 'a,b\n111111,-11111\n222222,-22222\n333333,-33333\n'}
    with filetexts(files):
        dd.read_csv('.overflow1.csv', sample=52)
        df = dd.read_csv('.overflow2.csv', sample=35)
        assert (df.dtypes == 'i8').all()