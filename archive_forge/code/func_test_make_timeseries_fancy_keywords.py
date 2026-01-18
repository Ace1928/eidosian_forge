from __future__ import annotations
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, get_string_dtype
def test_make_timeseries_fancy_keywords():
    df = dd.demo.make_timeseries('2000', '2001', {'A_B': int, 'B_': int, 'C': str}, freq='1D', partition_freq=f'6{ME}', A_B_lam=1000000, B__lam=2)
    a_cardinality = df.A_B.nunique()
    b_cardinality = df.B_.nunique()
    aa, bb = dask.compute(a_cardinality, b_cardinality, scheduler='single-threaded')
    assert 100 < aa <= 10000000
    assert 1 < bb <= 100