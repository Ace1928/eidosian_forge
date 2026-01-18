import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_duplicates_categorical_non_bool2_keeplast(self, cat_series):
    tc2 = cat_series
    expected = Series([False, True, True, False, False, False, False])
    result = tc2.duplicated(keep='last')
    tm.assert_series_equal(result, expected)
    result = tc2.drop_duplicates(keep='last')
    tm.assert_series_equal(result, tc2[~expected])
    sc = tc2.copy()
    return_value = sc.drop_duplicates(keep='last', inplace=True)
    assert return_value is None
    tm.assert_series_equal(sc, tc2[~expected])