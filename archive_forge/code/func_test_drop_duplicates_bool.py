import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('keep, expected', [('first', Series([False, False, True, True])), ('last', Series([True, True, False, False])), (False, Series([True, True, True, True]))])
def test_drop_duplicates_bool(keep, expected):
    tc = Series([True, False, True, False])
    tm.assert_series_equal(tc.duplicated(keep=keep), expected)
    tm.assert_series_equal(tc.drop_duplicates(keep=keep), tc[~expected])
    sc = tc.copy()
    return_value = sc.drop_duplicates(keep=keep, inplace=True)
    tm.assert_series_equal(sc, tc[~expected])
    assert return_value is None