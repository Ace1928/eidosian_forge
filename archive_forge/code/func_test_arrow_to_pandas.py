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
@PYARROW_MARK
def test_arrow_to_pandas(tmpdir, engine):
    df = pd.DataFrame({'A': [pd.Timestamp('2000-01-01')]})
    path = str(tmpdir.join('test.parquet'))
    df.to_parquet(path, engine=engine)
    arrow_to_pandas = {'timestamp_as_object': True}
    expect = pq.ParquetFile(path).read().to_pandas(**arrow_to_pandas)
    got = dd.read_parquet(path, engine='pyarrow', arrow_to_pandas=arrow_to_pandas)
    assert_eq(expect, got)
    assert got.A.dtype == got.compute().A.dtype