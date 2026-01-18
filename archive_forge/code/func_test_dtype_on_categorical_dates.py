from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_dtype_on_categorical_dates(self):
    df = DataFrame([[date(2001, 1, 1), 1.1], [date(2001, 1, 2), 1.3]], columns=['date', 'num2'])
    df['date'] = df['date'].astype('category')
    df2 = DataFrame([[date(2001, 1, 1), 1.3], [date(2001, 1, 3), 1.4]], columns=['date', 'num4'])
    df2['date'] = df2['date'].astype('category')
    expected_outer = DataFrame([[pd.Timestamp('2001-01-01').date(), 1.1, 1.3], [pd.Timestamp('2001-01-02').date(), 1.3, np.nan], [pd.Timestamp('2001-01-03').date(), np.nan, 1.4]], columns=['date', 'num2', 'num4'])
    result_outer = merge(df, df2, how='outer', on=['date'])
    tm.assert_frame_equal(result_outer, expected_outer)
    expected_inner = DataFrame([[pd.Timestamp('2001-01-01').date(), 1.1, 1.3]], columns=['date', 'num2', 'num4'])
    result_inner = merge(df, df2, how='inner', on=['date'])
    tm.assert_frame_equal(result_inner, expected_inner)