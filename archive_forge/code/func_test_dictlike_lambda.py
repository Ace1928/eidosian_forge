from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('ops, by_row, expected', [({'a': lambda x: x + 1}, 'compat', DataFrame({'a': [2, 3]})), ({'a': lambda x: x + 1}, False, DataFrame({'a': [2, 3]})), ({'a': lambda x: x.sum()}, 'compat', Series({'a': 3})), ({'a': lambda x: x.sum()}, False, Series({'a': 3})), ({'a': ['sum', np.sum, lambda x: x.sum()]}, 'compat', DataFrame({'a': [3, 3, 3]}, index=['sum', 'sum', '<lambda>'])), ({'a': ['sum', np.sum, lambda x: x.sum()]}, False, DataFrame({'a': [3, 3, 3]}, index=['sum', 'sum', '<lambda>'])), ({'a': lambda x: 1}, 'compat', DataFrame({'a': [1, 1]})), ({'a': lambda x: 1}, False, Series({'a': 1}))])
def test_dictlike_lambda(ops, by_row, expected):
    df = DataFrame({'a': [1, 2]})
    result = df.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)