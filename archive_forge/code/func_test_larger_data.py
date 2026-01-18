from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
def test_larger_data():
    df = dd.demo.make_timeseries('2000-01-01', '2000-04-01', {'value': float, 'id': int}, freq='10s', partition_freq='1D', seed=1)
    assert df.nunique_approx().compute() > 1000