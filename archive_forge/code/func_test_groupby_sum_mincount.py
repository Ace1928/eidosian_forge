import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
@pytest.mark.parametrize('min_count', [0, 10])
def test_groupby_sum_mincount(self, data_for_grouping, min_count):
    df = pd.DataFrame({'A': [1, 1, 2, 2, 3, 3, 1], 'B': data_for_grouping})
    result = df.groupby('A').sum(min_count=min_count)
    if min_count == 0:
        expected = pd.DataFrame({'B': pd.array([3, 0, 0], dtype='Int64')}, index=pd.Index([1, 2, 3], name='A'))
        tm.assert_frame_equal(result, expected)
    else:
        expected = pd.DataFrame({'B': pd.array([pd.NA] * 3, dtype='Int64')}, index=pd.Index([1, 2, 3], name='A'))
        tm.assert_frame_equal(result, expected)