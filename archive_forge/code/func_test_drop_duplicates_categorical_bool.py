import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_duplicates_categorical_bool(self, ordered):
    tc = Series(Categorical([True, False, True, False], categories=[True, False], ordered=ordered))
    expected = Series([False, False, True, True])
    tm.assert_series_equal(tc.duplicated(), expected)
    tm.assert_series_equal(tc.drop_duplicates(), tc[~expected])
    sc = tc.copy()
    return_value = sc.drop_duplicates(inplace=True)
    assert return_value is None
    tm.assert_series_equal(sc, tc[~expected])
    expected = Series([True, True, False, False])
    tm.assert_series_equal(tc.duplicated(keep='last'), expected)
    tm.assert_series_equal(tc.drop_duplicates(keep='last'), tc[~expected])
    sc = tc.copy()
    return_value = sc.drop_duplicates(keep='last', inplace=True)
    assert return_value is None
    tm.assert_series_equal(sc, tc[~expected])
    expected = Series([True, True, True, True])
    tm.assert_series_equal(tc.duplicated(keep=False), expected)
    tm.assert_series_equal(tc.drop_duplicates(keep=False), tc[~expected])
    sc = tc.copy()
    return_value = sc.drop_duplicates(keep=False, inplace=True)
    assert return_value is None
    tm.assert_series_equal(sc, tc[~expected])