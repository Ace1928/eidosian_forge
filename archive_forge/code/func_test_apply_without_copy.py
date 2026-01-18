from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_without_copy():
    data = DataFrame({'id_field': [100, 100, 200, 300], 'category': ['a', 'b', 'c', 'c'], 'value': [1, 2, 3, 4]})

    def filt1(x):
        if x.shape[0] == 1:
            return x.copy()
        else:
            return x[x.category == 'c']

    def filt2(x):
        if x.shape[0] == 1:
            return x
        else:
            return x[x.category == 'c']
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = data.groupby('id_field').apply(filt1)
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = data.groupby('id_field').apply(filt2)
    tm.assert_frame_equal(result, expected)