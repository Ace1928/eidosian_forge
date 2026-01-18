import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_duplicates_categorical_non_bool_keepfalse(self, cat_series_unused_category):
    tc1 = cat_series_unused_category
    expected = Series([False, False, True, True])
    result = tc1.duplicated(keep=False)
    tm.assert_series_equal(result, expected)
    result = tc1.drop_duplicates(keep=False)
    tm.assert_series_equal(result, tc1[~expected])
    sc = tc1.copy()
    return_value = sc.drop_duplicates(keep=False, inplace=True)
    assert return_value is None
    tm.assert_series_equal(sc, tc1[~expected])