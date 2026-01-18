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
def test_columns_no_index(tmpdir, write_engine, read_engine):
    fn = str(tmpdir)
    ddf.to_parquet(fn, engine=write_engine)
    ddf2 = ddf.reset_index()
    assert_eq(dd.read_parquet(fn, index=False, engine=read_engine, calculate_divisions=True), ddf2, check_index=False, check_divisions=True)
    assert_eq(dd.read_parquet(fn, index=False, columns=['x', 'y'], engine=read_engine, calculate_divisions=True), ddf2[['x', 'y']], check_index=False, check_divisions=True)
    assert_eq(dd.read_parquet(fn, index=False, columns=['myindex', 'x'], engine=read_engine, calculate_divisions=True), ddf2[['myindex', 'x']], check_index=False, check_divisions=True)