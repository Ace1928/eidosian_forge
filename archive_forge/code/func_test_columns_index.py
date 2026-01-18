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
def test_columns_index(tmpdir, write_engine, read_engine):
    fn = str(tmpdir)
    ddf.to_parquet(fn, engine=write_engine)
    assert_eq(dd.read_parquet(fn, columns=[], engine=read_engine, index='myindex', calculate_divisions=True), ddf[[]])
    assert_eq(dd.read_parquet(fn, columns=[], engine=read_engine, index='myindex', calculate_divisions=False), ddf[[]].clear_divisions(), check_divisions=True)
    assert_eq(dd.read_parquet(fn, index='myindex', columns=['x'], engine=read_engine, calculate_divisions=True), ddf[['x']])
    assert_eq(dd.read_parquet(fn, index='myindex', columns=['x'], engine=read_engine, calculate_divisions=False), ddf[['x']].clear_divisions(), check_divisions=True)
    assert_eq(dd.read_parquet(fn, index='myindex', columns=['x', 'y'], engine=read_engine, calculate_divisions=True), ddf)
    assert_eq(dd.read_parquet(fn, index='myindex', columns=['x', 'y'], engine=read_engine, calculate_divisions=False), ddf.clear_divisions(), check_divisions=True)