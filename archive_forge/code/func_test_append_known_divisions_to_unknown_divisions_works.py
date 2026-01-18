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
@pytest.mark.xfail(DASK_EXPR_ENABLED, reason='will be supported after string option')
def test_append_known_divisions_to_unknown_divisions_works(tmpdir, engine):
    tmp = str(tmpdir)
    df1 = pd.DataFrame({'x': np.arange(100), 'y': np.arange(100, 200)}, index=np.arange(100, 0, -1))
    ddf1 = dd.from_pandas(df1, npartitions=3, sort=False)
    df2 = pd.DataFrame({'x': np.arange(100, 200), 'y': np.arange(200, 300)})
    df2.index = df2.index.astype(df1.index.dtype)
    ddf2 = dd.from_pandas(df2, npartitions=3)
    ddf1.to_parquet(tmp, engine=engine, write_metadata_file=True)
    ddf2.to_parquet(tmp, engine=engine, append=True)
    res = dd.read_parquet(tmp, engine=engine)
    sol = pd.concat([df1, df2])
    assert_eq(res, sol)