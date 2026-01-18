from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('function, expected_values', [(lambda x: x.index.to_list(), [[0, 1], [2, 3]]), (lambda x: set(x.index.to_list()), [{0, 1}, {2, 3}]), (lambda x: tuple(x.index.to_list()), [(0, 1), (2, 3)]), (lambda x: dict(enumerate(x.index.to_list())), [{0: 0, 1: 1}, {0: 2, 1: 3}]), (lambda x: [{n: i} for n, i in enumerate(x.index.to_list())], [[{0: 0}, {1: 1}], [{0: 2}, {1: 3}]])])
def test_apply_function_returns_non_pandas_non_scalar(function, expected_values):
    df = DataFrame(['A', 'A', 'B', 'B'], columns=['groups'])
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('groups').apply(function)
    expected = Series(expected_values, index=Index(['A', 'B'], name='groups'))
    tm.assert_series_equal(result, expected)