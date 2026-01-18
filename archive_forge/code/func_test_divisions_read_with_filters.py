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
@FASTPARQUET_MARK
@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_divisions_read_with_filters(tmpdir):
    tmpdir = str(tmpdir)
    size = 100
    categoricals = []
    for value in ['a', 'b', 'c', 'd']:
        categoricals += [value] * int(size / 4)
    df = pd.DataFrame({'a': categoricals, 'b': np.random.random(size=size), 'c': np.random.randint(1, 5, size=size)})
    d = dd.from_pandas(df, npartitions=4)
    d.to_parquet(tmpdir, write_index=True, partition_on=['a'], engine='fastparquet')
    out = dd.read_parquet(tmpdir, engine='fastparquet', filters=[('a', '==', 'b')], calculate_divisions=True)
    expected_divisions = (25, 49)
    assert out.divisions == expected_divisions