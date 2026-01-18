from datetime import datetime
import numpy as np
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_concat_dtypes(self, using_infer_string):
    index = ['cat', 'obj', 'num']
    cat = Categorical(['a', 'b', 'c'])
    obj = Series(['a', 'b', 'c'])
    num = Series([1, 2, 3])
    df = pd.concat([Series(cat), obj, num], axis=1, keys=index)
    result = df.dtypes == (object if not using_infer_string else 'string[pyarrow_numpy]')
    expected = Series([False, True, False], index=index)
    tm.assert_series_equal(result, expected)
    result = df.dtypes == 'int64'
    expected = Series([False, False, True], index=index)
    tm.assert_series_equal(result, expected)
    result = df.dtypes == 'category'
    expected = Series([True, False, False], index=index)
    tm.assert_series_equal(result, expected)