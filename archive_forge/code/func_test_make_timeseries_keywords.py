from __future__ import annotations
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, get_string_dtype
def test_make_timeseries_keywords():
    df = dd.demo.make_timeseries('2000', '2001', {'A': int, 'B': int, 'C': str}, freq='1D', partition_freq=f'6{ME}', A_lam=1000000, B_lam=2)
    a_cardinality = df.A.nunique()
    b_cardinality = df.B.nunique()
    aa, bb = dask.compute(a_cardinality, b_cardinality, scheduler='single-threaded')
    assert 100 < aa <= 10000000
    assert 1 < bb <= 100