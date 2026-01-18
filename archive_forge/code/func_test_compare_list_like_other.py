import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.tests.arithmetic.common import get_upcast_box
@pytest.mark.parametrize('other', [np.arange(4, dtype='int64'), np.arange(4, dtype='float64'), date_range('2017-01-01', periods=4), date_range('2017-01-01', periods=4, tz='US/Eastern'), timedelta_range('0 days', periods=4), period_range('2017-01-01', periods=4, freq='D'), Categorical(list('abab')), Categorical(date_range('2017-01-01', periods=4)), pd.array(list('abcd')), pd.array(['foo', 3.14, None, object()], dtype=object)], ids=lambda x: str(x.dtype))
def test_compare_list_like_other(self, op, interval_array, other):
    result = op(interval_array, other)
    expected = self.elementwise_comparison(op, interval_array, other)
    tm.assert_numpy_array_equal(result, expected)