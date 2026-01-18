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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason="doesn't make sense")
@pytest.mark.parametrize('fs', ['fsspec', None])
def test_filesystem_option(tmp_path, engine, fs):
    from fsspec.implementations.local import LocalFileSystem
    df = pd.DataFrame({'a': range(10)})
    dd.from_pandas(df, npartitions=2).to_parquet(tmp_path, engine=engine)
    filesystem = fs or LocalFileSystem()
    ddf = dd.read_parquet(tmp_path, engine=engine, filesystem=filesystem)
    if fs is None:
        layer_fs = next(iter(ddf.dask.layers.values())).io_func.fs
        assert layer_fs is filesystem
    assert_eq(ddf, df)