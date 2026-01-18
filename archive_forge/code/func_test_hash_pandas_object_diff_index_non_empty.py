import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
@pytest.mark.parametrize('obj', [Series([1, 2, 3]), Series([1.0, 1.5, 3.2]), Series([1.0, 1.5, np.nan]), Series([1.0, 1.5, 3.2], index=[1.5, 1.1, 3.3]), Series(['a', 'b', 'c']), Series(['a', np.nan, 'c']), Series(['a', None, 'c']), Series([True, False, True]), DataFrame({'x': ['a', 'b', 'c'], 'y': [1, 2, 3]}), DataFrame(np.full((10, 4), np.nan)), DataFrame({'A': [0.0, 1.0, 2.0, 3.0, 4.0], 'B': [0.0, 1.0, 0.0, 1.0, 0.0], 'C': Index(['foo1', 'foo2', 'foo3', 'foo4', 'foo5'], dtype=object), 'D': pd.date_range('20130101', periods=5)}), DataFrame(range(5), index=pd.date_range('2020-01-01', periods=5)), Series(range(5), index=pd.date_range('2020-01-01', periods=5)), Series(period_range('2020-01-01', periods=10, freq='D')), Series(pd.date_range('20130101', periods=3, tz='US/Eastern'))])
def test_hash_pandas_object_diff_index_non_empty(obj):
    a = hash_pandas_object(obj, index=True)
    b = hash_pandas_object(obj, index=False)
    assert not (a == b).all()