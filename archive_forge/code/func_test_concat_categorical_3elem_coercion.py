import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_categorical_3elem_coercion(self):
    s1 = Series([1, 2, np.nan], dtype='category')
    s2 = Series([2, 1, 2], dtype='category')
    s3 = Series([1, 2, 1, 2, np.nan])
    exp = Series([1, 2, np.nan, 2, 1, 2, 1, 2, 1, 2, np.nan], dtype='float')
    tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)
    tm.assert_series_equal(s1._append([s2, s3], ignore_index=True), exp)
    exp = Series([1, 2, 1, 2, np.nan, 1, 2, np.nan, 2, 1, 2], dtype='float')
    tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)
    tm.assert_series_equal(s3._append([s1, s2], ignore_index=True), exp)
    s1 = Series([4, 5, 6], dtype='category')
    s2 = Series([1, 2, 3], dtype='category')
    s3 = Series([1, 3, 4])
    exp = Series([4, 5, 6, 1, 2, 3, 1, 3, 4])
    tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)
    tm.assert_series_equal(s1._append([s2, s3], ignore_index=True), exp)
    exp = Series([1, 3, 4, 4, 5, 6, 1, 2, 3])
    tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)
    tm.assert_series_equal(s3._append([s1, s2], ignore_index=True), exp)
    s1 = Series([4, 5, 6], dtype='category')
    s2 = Series([1, 2, 3], dtype='category')
    s3 = Series([10, 11, 12])
    exp = Series([4, 5, 6, 1, 2, 3, 10, 11, 12])
    tm.assert_series_equal(pd.concat([s1, s2, s3], ignore_index=True), exp)
    tm.assert_series_equal(s1._append([s2, s3], ignore_index=True), exp)
    exp = Series([10, 11, 12, 4, 5, 6, 1, 2, 3])
    tm.assert_series_equal(pd.concat([s3, s1, s2], ignore_index=True), exp)
    tm.assert_series_equal(s3._append([s1, s2], ignore_index=True), exp)