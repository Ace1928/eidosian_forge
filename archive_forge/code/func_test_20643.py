from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_20643():
    orig = Series([0, 1, 2], index=['a', 'b', 'c'])
    expected = Series([0, 2.7, 2], index=['a', 'b', 'c'])
    ser = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        ser.at['b'] = 2.7
    tm.assert_series_equal(ser, expected)
    ser = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        ser.loc['b'] = 2.7
    tm.assert_series_equal(ser, expected)
    ser = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        ser['b'] = 2.7
    tm.assert_series_equal(ser, expected)
    ser = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        ser.iat[1] = 2.7
    tm.assert_series_equal(ser, expected)
    ser = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        ser.iloc[1] = 2.7
    tm.assert_series_equal(ser, expected)
    orig_df = orig.to_frame('A')
    expected_df = expected.to_frame('A')
    df = orig_df.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        df.at['b', 'A'] = 2.7
    tm.assert_frame_equal(df, expected_df)
    df = orig_df.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        df.loc['b', 'A'] = 2.7
    tm.assert_frame_equal(df, expected_df)
    df = orig_df.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        df.iloc[1, 0] = 2.7
    tm.assert_frame_equal(df, expected_df)
    df = orig_df.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        df.iat[1, 0] = 2.7
    tm.assert_frame_equal(df, expected_df)