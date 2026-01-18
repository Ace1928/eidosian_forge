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
@pytest.mark.parametrize('partition_on', ('b', None))
def test_extra_file(tmpdir, engine, partition_on):
    tmpdir = str(tmpdir)
    df = pd.DataFrame({'a': range(100), 'b': ['dog', 'cat'] * 50})
    df = df.assign(b=df.b.astype('category'))
    ddf = dd.from_pandas(df, npartitions=2)
    ddf.to_parquet(tmpdir, engine=engine, write_metadata_file=True, partition_on=partition_on)
    open(os.path.join(tmpdir, '_SUCCESS'), 'w').close()
    open(os.path.join(tmpdir, 'part.0.parquet.crc'), 'w').close()
    os.remove(os.path.join(tmpdir, '_metadata'))
    out = dd.read_parquet(tmpdir, engine=engine, calculate_divisions=True)
    assert_eq(out, df, check_categorical=False)
    assert_eq(out.b, df.b, check_category_order=False)

    def _parquet_file_extension(val, legacy=False):
        return {'dataset': {'require_extension': val}} if legacy else {'parquet_file_extension': val}
    out = dd.read_parquet(tmpdir, engine=engine, **_parquet_file_extension('.parquet'), calculate_divisions=True)
    assert_eq(out, df, check_categorical=False)
    assert_eq(out.b, df.b, check_category_order=False)
    if not DASK_EXPR_ENABLED:
        with pytest.warns(FutureWarning, match='require_extension is deprecated'):
            out = dd.read_parquet(tmpdir, engine=engine, **_parquet_file_extension('.parquet', legacy=True), calculate_divisions=True)
    with pytest.raises((OSError, pa.lib.ArrowInvalid)):
        dd.read_parquet(tmpdir, engine=engine, **_parquet_file_extension(None)).compute()
    with pytest.raises(ValueError):
        dd.read_parquet(tmpdir, engine=engine, **_parquet_file_extension('.foo')).compute()