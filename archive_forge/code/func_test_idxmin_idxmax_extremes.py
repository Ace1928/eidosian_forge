import builtins
import datetime as dt
from string import ascii_lowercase
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
@pytest.mark.parametrize('how', ['idxmin', 'idxmax'])
def test_idxmin_idxmax_extremes(how, any_real_numpy_dtype):
    if any_real_numpy_dtype is int or any_real_numpy_dtype is float:
        return
    info = np.iinfo if 'int' in any_real_numpy_dtype else np.finfo
    min_value = info(any_real_numpy_dtype).min
    max_value = info(any_real_numpy_dtype).max
    df = DataFrame({'a': [2, 1, 1, 2], 'b': [min_value, max_value, max_value, min_value]}, dtype=any_real_numpy_dtype)
    gb = df.groupby('a')
    result = getattr(gb, how)()
    expected = DataFrame({'b': [1, 0]}, index=pd.Index([1, 2], name='a', dtype=any_real_numpy_dtype))
    tm.assert_frame_equal(result, expected)