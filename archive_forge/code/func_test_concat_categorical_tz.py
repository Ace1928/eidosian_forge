from datetime import datetime
import numpy as np
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_categorical_tz(self):
    a = Series(pd.date_range('2017-01-01', periods=2, tz='US/Pacific'))
    b = Series(['a', 'b'], dtype='category')
    result = pd.concat([a, b], ignore_index=True)
    expected = Series([pd.Timestamp('2017-01-01', tz='US/Pacific'), pd.Timestamp('2017-01-02', tz='US/Pacific'), 'a', 'b'])
    tm.assert_series_equal(result, expected)