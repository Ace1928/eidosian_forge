from datetime import datetime
import numpy as np
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_categorical_datetime(self):
    df1 = DataFrame({'x': Series(datetime(2021, 1, 1), index=[0], dtype='category')})
    df2 = DataFrame({'x': Series(datetime(2021, 1, 2), index=[1], dtype='category')})
    result = pd.concat([df1, df2])
    expected = DataFrame({'x': Series([datetime(2021, 1, 1), datetime(2021, 1, 2)])})
    tm.assert_equal(result, expected)