from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('scalar1, scalar2', [(datetime(2020, 1, 1), datetime(2020, 1, 2)), (pd.Period('2020-01-01', 'D'), pd.Period('2020-01-02', 'D')), (pd.Timedelta('89 days'), pd.Timedelta('60 min')), (pd.Interval(left=0, right=1), pd.Interval(left=2, right=3, closed='left'))])
def test_combine_first_timestamp_bug(scalar1, scalar2, nulls_fixture):
    na_value = nulls_fixture
    frame = DataFrame([[na_value, na_value]], columns=['a', 'b'])
    other = DataFrame([[scalar1, scalar2]], columns=['b', 'c'])
    common_dtype = find_common_type([frame.dtypes['b'], other.dtypes['b']])
    if is_dtype_equal(common_dtype, 'object') or frame.dtypes['b'] == other.dtypes['b']:
        val = scalar1
    else:
        val = na_value
    result = frame.combine_first(other)
    expected = DataFrame([[na_value, val, scalar2]], columns=['a', 'b', 'c'])
    expected['b'] = expected['b'].astype(common_dtype)
    tm.assert_frame_equal(result, expected)