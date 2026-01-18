import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_categorical_empty(self):
    s1 = Series([], dtype='category')
    s2 = Series([1, 2], dtype='category')
    msg = 'The behavior of array concatenation with empty entries is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), s2)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), s2)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), s2)
        tm.assert_series_equal(s2._append(s1, ignore_index=True), s2)
    s1 = Series([], dtype='category')
    s2 = Series([], dtype='category')
    tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), s2)
    tm.assert_series_equal(s1._append(s2, ignore_index=True), s2)
    s1 = Series([], dtype='category')
    s2 = Series([], dtype='object')
    tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), s2)
    tm.assert_series_equal(s1._append(s2, ignore_index=True), s2)
    tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), s2)
    tm.assert_series_equal(s2._append(s1, ignore_index=True), s2)
    s1 = Series([], dtype='category')
    s2 = Series([np.nan, np.nan])
    exp = Series([np.nan, np.nan])
    with tm.assert_produces_warning(FutureWarning, match=msg):
        tm.assert_series_equal(pd.concat([s1, s2], ignore_index=True), exp)
        tm.assert_series_equal(s1._append(s2, ignore_index=True), exp)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        tm.assert_series_equal(pd.concat([s2, s1], ignore_index=True), exp)
        tm.assert_series_equal(s2._append(s1, ignore_index=True), exp)