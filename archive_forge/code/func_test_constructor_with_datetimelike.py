from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtl', [date_range('1995-01-01 00:00:00', periods=5, freq='s'), date_range('1995-01-01 00:00:00', periods=5, freq='s', tz='US/Eastern'), timedelta_range('1 day', periods=5, freq='s')])
def test_constructor_with_datetimelike(self, dtl):
    s = Series(dtl)
    c = Categorical(s)
    expected = type(dtl)(s)
    expected._data.freq = None
    tm.assert_index_equal(c.categories, expected)
    tm.assert_numpy_array_equal(c.codes, np.arange(5, dtype='int8'))
    s2 = s.copy()
    s2.iloc[-1] = NaT
    c = Categorical(s2)
    expected = type(dtl)(s2.dropna())
    expected._data.freq = None
    tm.assert_index_equal(c.categories, expected)
    exp = np.array([0, 1, 2, 3, -1], dtype=np.int8)
    tm.assert_numpy_array_equal(c.codes, exp)
    result = repr(c)
    assert 'NaT' in result