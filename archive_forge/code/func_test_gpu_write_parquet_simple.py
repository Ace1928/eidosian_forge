from __future__ import annotations
import contextlib
import glob
import math
import os
import sys
import warnings
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask
import dask.dataframe as dd
import dask.multiprocessing
from dask.array.numpy_compat import NUMPY_GE_124
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import (
from dask.dataframe.io.parquet.core import get_engine
from dask.dataframe.io.parquet.utils import _parse_pandas_metadata
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
from dask.layers import DataFrameIOLayer
from dask.utils import natural_sort_key
from dask.utils_test import hlg_layer
@pytest.mark.gpu
def test_gpu_write_parquet_simple(tmpdir):
    fn = str(tmpdir)
    cudf = pytest.importorskip('cudf')
    dask_cudf = pytest.importorskip('dask_cudf')
    from dask.dataframe.dispatch import pyarrow_schema_dispatch

    @pyarrow_schema_dispatch.register((cudf.DataFrame,))
    def get_pyarrow_schema_cudf(obj):
        return obj.to_arrow().schema
    df = cudf.DataFrame({'a': ['abc', 'def'], 'b': ['a', 'z']})
    ddf = dask_cudf.from_cudf(df, 3)
    ddf.to_parquet(fn)
    got = dask_cudf.read_parquet(fn)
    assert_eq(df, got)