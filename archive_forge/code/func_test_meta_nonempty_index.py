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
def test_meta_nonempty_index():
    idx = pd.RangeIndex(1, name='foo')
    res = meta_nonempty(idx)
    assert type(res) is pd.RangeIndex
    assert res.name == idx.name
    idx = pd.Index([1], name='foo', dtype='int')
    res = meta_nonempty(idx)
    assert type(res) is type(idx)
    if PANDAS_GE_200:
        assert res.dtype == np.int_
    else:
        assert res.dtype == 'int64'
    assert res.name == idx.name
    idx = pd.Index(['a'], name='foo')
    res = meta_nonempty(idx)
    assert type(res) is pd.Index
    assert res.name == idx.name
    idx = pd.DatetimeIndex(['1970-01-01'], freq='d', tz='America/New_York', name='foo')
    res = meta_nonempty(idx)
    assert type(res) is pd.DatetimeIndex
    assert res.tz == idx.tz
    assert res.freq == idx.freq
    assert res.name == idx.name
    idx = pd.PeriodIndex(['1970-01-01'], freq='d', name='foo')
    res = meta_nonempty(idx)
    assert type(res) is pd.PeriodIndex
    assert res.freq == idx.freq
    assert res.name == idx.name
    idx = pd.TimedeltaIndex([pd.Timedelta(1, 'D')], freq='d', name='foo')
    res = meta_nonempty(idx)
    assert type(res) is pd.TimedeltaIndex
    assert res.freq == idx.freq
    assert res.name == idx.name
    idx = pd.CategoricalIndex(['xyx'], ['xyx', 'zzz'], ordered=True, name='foo')
    res = meta_nonempty(idx)
    assert type(res) is pd.CategoricalIndex
    assert (res.categories == idx.categories).all()
    assert res.ordered == idx.ordered
    assert res.name == idx.name
    idx = pd.CategoricalIndex([], [UNKNOWN_CATEGORIES], ordered=True, name='foo')
    res = meta_nonempty(idx)
    assert type(res) is pd.CategoricalIndex
    assert res.ordered == idx.ordered
    assert res.name == idx.name
    levels = [pd.Index([1], name='a'), pd.Index([1.0], name='b')]
    codes = [[0], [0]]
    idx = pd.MultiIndex(levels=levels, names=['a', 'b'], codes=codes)
    res = meta_nonempty(idx)
    assert type(res) is pd.MultiIndex
    for idx1, idx2 in zip(idx.levels, res.levels):
        assert type(idx1) is type(idx2)
        assert idx1.name == idx2.name
    assert res.names == idx.names
    levels = [pd.Index([1], name='a'), pd.CategoricalIndex(data=['xyx'], categories=['xyx'], name='b'), pd.TimedeltaIndex([np.timedelta64(1, 'D')], name='timedelta')]
    codes = [[0], [0], [0]]
    idx = pd.MultiIndex(levels=levels, names=['a', 'b', 'timedelta'], codes=codes)
    res = meta_nonempty(idx)
    assert type(res) is pd.MultiIndex
    for idx1, idx2 in zip(idx.levels, res.levels):
        assert type(idx1) is type(idx2)
        assert idx1.name == idx2.name
    assert res.names == idx.names