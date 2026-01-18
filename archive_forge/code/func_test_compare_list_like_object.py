import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.tests.arithmetic.common import get_upcast_box
@pytest.mark.parametrize('other', [(Interval(0, 1), Interval(Timedelta('1 day'), Timedelta('2 days')), Interval(4, 5, 'both'), Interval(10, 20, 'neither')), (0, 1.5, Timestamp('20170103'), np.nan), (Timestamp('20170102', tz='US/Eastern'), Timedelta('2 days'), 'baz', pd.NaT)])
def test_compare_list_like_object(self, op, interval_array, other):
    result = op(interval_array, other)
    expected = self.elementwise_comparison(op, interval_array, other)
    tm.assert_numpy_array_equal(result, expected)