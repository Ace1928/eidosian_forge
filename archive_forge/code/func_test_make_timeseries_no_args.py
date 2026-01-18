from __future__ import annotations
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, get_string_dtype
def test_make_timeseries_no_args():
    df = dd.demo.make_timeseries()
    assert 1 < df.npartitions < 1000
    assert len(df.columns) > 1
    assert len(set(df.dtypes)) > 1