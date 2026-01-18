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
@pytest.mark.skip_with_pyarrow_strings
def test_getitem_optimization_after_filter():
    with filetext(timeseries) as fn:
        expect = pd.read_csv(fn)
        expect = expect[expect['High'] > 205.0][['Low']]
        ddf = dd.read_csv(fn)
        ddf = ddf[ddf['High'] > 205.0][['Low']]
        dsk = optimize_dataframe_getitem(ddf.dask, keys=[ddf._name])
        subgraph_rd = hlg_layer(dsk, 'read-csv')
        assert isinstance(subgraph_rd, DataFrameIOLayer)
        assert set(subgraph_rd.columns) == {'High', 'Low'}
        assert_eq(expect, ddf)