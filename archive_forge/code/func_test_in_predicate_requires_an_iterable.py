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
@pytest.mark.parametrize('filter_value', ([('B', 'in', 10)], [[('B', 'in', 10)]], [('B', '<', 10), ('B', 'in', 10)], [[('B', '<', 10), ('B', 'in', 10)]]), ids=('one-item-single-nest', 'one-item-double-nest', 'two-item-double-nest', 'two-item-two-nest'))
def test_in_predicate_requires_an_iterable(tmp_path, engine, filter_value):
    """Regression test for https://github.com/dask/dask/issues/8720"""
    path = tmp_path / 'gh_8720_pandas.parquet'
    df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [1, 1, 2, 2]})
    df.to_parquet(path, engine=engine)
    with pytest.raises(TypeError, match="Value of 'in' filter"):
        dd.read_parquet(path, engine=engine, filters=filter_value)
    ddf = dd.from_pandas(df, npartitions=2)
    path = tmp_path / 'gh_8720_dask.parquet'
    ddf.to_parquet(path, engine=engine)
    with pytest.raises(TypeError, match="Value of 'in' filter"):
        dd.read_parquet(path, engine=engine, filters=filter_value)