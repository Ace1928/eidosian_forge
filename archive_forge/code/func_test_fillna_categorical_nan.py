import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_categorical_nan(self):
    cat = Categorical([np.nan, 2, np.nan])
    val = Categorical([np.nan, np.nan, np.nan])
    df = DataFrame({'cats': cat, 'vals': val})
    median = Series({'cats': 2.0, 'vals': np.nan})
    res = df.fillna(median)
    v_exp = [np.nan, np.nan, np.nan]
    df_exp = DataFrame({'cats': [2, 2, 2], 'vals': v_exp}, dtype='category')
    tm.assert_frame_equal(res, df_exp)
    result = df.cats.fillna(np.nan)
    tm.assert_series_equal(result, df.cats)
    result = df.vals.fillna(np.nan)
    tm.assert_series_equal(result, df.vals)
    idx = DatetimeIndex(['2011-01-01 09:00', '2016-01-01 23:45', '2011-01-01 09:00', NaT, NaT])
    df = DataFrame({'a': Categorical(idx)})
    tm.assert_frame_equal(df.fillna(value=NaT), df)
    idx = PeriodIndex(['2011-01', '2011-01', '2011-01', NaT, NaT], freq='M')
    df = DataFrame({'a': Categorical(idx)})
    tm.assert_frame_equal(df.fillna(value=NaT), df)
    idx = TimedeltaIndex(['1 days', '2 days', '1 days', NaT, NaT])
    df = DataFrame({'a': Categorical(idx)})
    tm.assert_frame_equal(df.fillna(value=NaT), df)