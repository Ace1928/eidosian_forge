import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('index', [pd.CategoricalIndex(list('abc')), pd.interval_range(0, 3), pd.period_range('2020', periods=3, freq='D'), MultiIndex.from_tuples([('a', 0), ('a', 1), ('b', 0)])])
def test_agg_index_has_complex_internals(index):
    df = DataFrame({'group': [1, 1, 2], 'value': [0, 1, 0]}, index=index)
    result = df.groupby('group').agg({'value': Series.nunique})
    expected = DataFrame({'group': [1, 2], 'value': [2, 1]}).set_index('group')
    tm.assert_frame_equal(result, expected)