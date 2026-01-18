from __future__ import annotations
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, get_string_dtype
def test_no_overlaps():
    df = dd.demo.make_timeseries('2000', '2001', {'A': float}, freq='3h', partition_freq=f'3{ME}')
    assert all((df.get_partition(i).index.max().compute() < df.get_partition(i + 1).index.min().compute() for i in range(df.npartitions - 2)))