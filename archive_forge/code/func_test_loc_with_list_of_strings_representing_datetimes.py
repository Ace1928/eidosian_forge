import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx,labels,expected_idx', [(period_range(start='2000', periods=20, freq='D'), ['2000-01-04', '2000-01-08', '2000-01-12'], [Period('2000-01-04', freq='D'), Period('2000-01-08', freq='D'), Period('2000-01-12', freq='D')]), (date_range(start='2000', periods=20, freq='D'), ['2000-01-04', '2000-01-08', '2000-01-12'], [Timestamp('2000-01-04'), Timestamp('2000-01-08'), Timestamp('2000-01-12')]), (pd.timedelta_range(start='1 day', periods=20), ['4D', '8D', '12D'], [pd.Timedelta('4 day'), pd.Timedelta('8 day'), pd.Timedelta('12 day')])])
def test_loc_with_list_of_strings_representing_datetimes(self, idx, labels, expected_idx, frame_or_series):
    obj = frame_or_series(range(20), index=idx)
    expected_value = [3, 7, 11]
    expected = frame_or_series(expected_value, expected_idx)
    tm.assert_equal(expected, obj.loc[labels])
    if frame_or_series is Series:
        tm.assert_series_equal(expected, obj[labels])