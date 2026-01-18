from datetime import datetime
import numpy as np
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_concat(self, sort):
    df1 = DataFrame(np.arange(18, dtype='int64').reshape(6, 3), columns=['a', 'b', 'c'])
    df2 = DataFrame(np.arange(14, dtype='int64').reshape(7, 2), columns=['a', 'c'])
    cat_values = ['one', 'one', 'two', 'one', 'two', 'two', 'one']
    df2['h'] = Series(Categorical(cat_values))
    res = pd.concat((df1, df2), axis=0, ignore_index=True, sort=sort)
    exp = DataFrame({'a': [0, 3, 6, 9, 12, 15, 0, 2, 4, 6, 8, 10, 12], 'b': [1, 4, 7, 10, 13, 16, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], 'c': [2, 5, 8, 11, 14, 17, 1, 3, 5, 7, 9, 11, 13], 'h': [None] * 6 + cat_values})
    exp['h'] = exp['h'].astype(df2['h'].dtype)
    tm.assert_frame_equal(res, exp)