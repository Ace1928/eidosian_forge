from __future__ import annotations
import re
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_300, tm
from dask.dataframe.core import apply_and_enforce
from dask.dataframe.utils import (
from dask.local import get_sync
@pytest.mark.parametrize('frame_value_counts', [True, False])
def test_is_dataframe_like(monkeypatch, frame_value_counts):
    if frame_value_counts:
        monkeypatch.setattr(pd.DataFrame, 'value_counts', lambda x: None, raising=False)
    df = pd.DataFrame({'x': [1, 2, 3]})
    ddf = dd.from_pandas(df, npartitions=1)
    assert is_dataframe_like(df)
    assert is_dataframe_like(ddf)
    assert not is_dataframe_like(df.x)
    assert not is_dataframe_like(ddf.x)
    assert not is_dataframe_like(df.index)
    assert not is_dataframe_like(ddf.index)
    assert not is_dataframe_like(pd.DataFrame)
    assert not is_series_like(df)
    assert not is_series_like(ddf)
    assert is_series_like(df.x)
    assert is_series_like(ddf.x)
    assert not is_series_like(df.index)
    assert not is_series_like(ddf.index)
    assert not is_series_like(pd.Series)
    assert not is_index_like(df)
    assert not is_index_like(ddf)
    assert not is_index_like(df.x)
    assert not is_index_like(ddf.x)
    assert is_index_like(df.index)
    assert is_index_like(ddf.index)
    assert not is_index_like(pd.Index)

    class DataFrameWrapper:
        __class__ = pd.DataFrame
    wrap = DataFrameWrapper()
    wrap.dtypes = None
    wrap.columns = None
    assert is_dataframe_like(wrap)

    class SeriesWrapper:
        __class__ = pd.Series
    wrap = SeriesWrapper()
    wrap.dtype = None
    wrap.name = None
    assert is_series_like(wrap)

    class IndexWrapper:
        __class__ = pd.Index
    wrap = IndexWrapper()
    wrap.dtype = None
    wrap.name = None
    assert is_index_like(wrap)