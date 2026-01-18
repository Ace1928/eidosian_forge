from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('index_values', [[1, 2, 3], [1.0, 2.0, 3.0]])
def test_apply_index_key_error_bug(index_values):
    result = DataFrame({'a': ['aa', 'a2', 'a3'], 'b': [1, 2, 3]}, index=Index(index_values))
    expected = DataFrame({'b_mean': [2.0, 3.0, 1.0]}, index=Index(['a2', 'a3', 'aa'], name='a'))
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = result.groupby('a').apply(lambda df: Series([df['b'].mean()], index=['b_mean']))
    tm.assert_frame_equal(result, expected)