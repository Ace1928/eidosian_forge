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
def test_filters_file_list(tmpdir, engine):
    df = pd.DataFrame({'x': range(10), 'y': list('aabbccddee')})
    ddf = dd.from_pandas(df, npartitions=5)
    ddf.to_parquet(str(tmpdir), engine=engine)
    files = str(tmpdir.join('*.parquet'))
    ddf_out = dd.read_parquet(files, calculate_divisions=True, engine=engine, filters=[('x', '>', 3)])
    assert ddf_out.npartitions == 3
    assert_eq(df[df['x'] > 3], ddf_out.compute(), check_index=False)
    ddf2 = dd.read_parquet(str(tmpdir.join('part.0.parquet')), calculate_divisions=True, engine=engine, filters=[('x', '>', 3)])
    assert len(ddf2) == 0
    pd.read_parquet(os.path.join(tmpdir, 'part.4.parquet'), engine=engine)[reversed(df.columns)].to_parquet(os.path.join(tmpdir, 'part.4.parquet'), engine=engine)
    ddf3 = dd.read_parquet(str(tmpdir.join('*.parquet')), calculate_divisions=True, engine=engine, filters=[('x', '>', 3)])
    assert ddf3.npartitions == 3
    assert_eq(df[df['x'] > 3], ddf3, check_index=False)