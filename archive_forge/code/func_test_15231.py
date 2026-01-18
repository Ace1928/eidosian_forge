import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(reason='unwanted upcast')
def test_15231():
    df = DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
    df.loc[2] = Series({'a': 5, 'b': 6})
    assert (df.dtypes == np.int64).all()
    df.loc[3] = Series({'a': 7})
    exp_dtypes = Series([np.int64, np.float64], dtype=object, index=['a', 'b'])
    tm.assert_series_equal(df.dtypes, exp_dtypes)