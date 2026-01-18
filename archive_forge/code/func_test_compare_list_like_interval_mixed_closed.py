import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.tests.arithmetic.common import get_upcast_box
def test_compare_list_like_interval_mixed_closed(self, op, interval_constructor, closed, other_closed):
    interval_array = IntervalArray.from_arrays(range(2), range(1, 3), closed=closed)
    other = interval_constructor(range(2), range(1, 3), closed=other_closed)
    result = op(interval_array, other)
    expected = self.elementwise_comparison(op, interval_array, other)
    tm.assert_equal(result, expected)