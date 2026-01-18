from datetime import datetime
import numpy as np
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_concat_preserve(self):
    s = Series(list('abc'), dtype='category')
    s2 = Series(list('abd'), dtype='category')
    exp = Series(list('abcabd'))
    res = pd.concat([s, s2], ignore_index=True)
    tm.assert_series_equal(res, exp)
    exp = Series(list('abcabc'), dtype='category')
    res = pd.concat([s, s], ignore_index=True)
    tm.assert_series_equal(res, exp)
    exp = Series(list('abcabc'), index=[0, 1, 2, 0, 1, 2], dtype='category')
    res = pd.concat([s, s])
    tm.assert_series_equal(res, exp)
    a = Series(np.arange(6, dtype='int64'))
    b = Series(list('aabbca'))
    df2 = DataFrame({'A': a, 'B': b.astype(CategoricalDtype(list('cab')))})
    res = pd.concat([df2, df2])
    exp = DataFrame({'A': pd.concat([a, a]), 'B': pd.concat([b, b]).astype(CategoricalDtype(list('cab')))})
    tm.assert_frame_equal(res, exp)