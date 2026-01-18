from datetime import timedelta
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_empty_frame_dtypes(self):
    empty_df = DataFrame()
    tm.assert_series_equal(empty_df.dtypes, Series(dtype=object))
    nocols_df = DataFrame(index=[1, 2, 3])
    tm.assert_series_equal(nocols_df.dtypes, Series(dtype=object))
    norows_df = DataFrame(columns=list('abc'))
    tm.assert_series_equal(norows_df.dtypes, Series(object, index=list('abc')))
    norows_int_df = DataFrame(columns=list('abc')).astype(np.int32)
    tm.assert_series_equal(norows_int_df.dtypes, Series(np.dtype('int32'), index=list('abc')))
    df = DataFrame({'a': 1, 'b': True, 'c': 1.0}, index=[1, 2, 3])
    ex_dtypes = Series({'a': np.int64, 'b': np.bool_, 'c': np.float64})
    tm.assert_series_equal(df.dtypes, ex_dtypes)
    tm.assert_series_equal(df[:0].dtypes, ex_dtypes)