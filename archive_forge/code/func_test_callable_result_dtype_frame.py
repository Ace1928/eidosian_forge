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
@pytest.mark.parametrize('keys, agg_index', [(['a'], Index([1], name='a')), (['a', 'b'], MultiIndex([[1], [2]], [[0], [0]], names=['a', 'b']))])
@pytest.mark.parametrize('input_dtype', ['bool', 'int32', 'int64', 'float32', 'float64'])
@pytest.mark.parametrize('result_dtype', ['bool', 'int32', 'int64', 'float32', 'float64'])
@pytest.mark.parametrize('method', ['apply', 'aggregate', 'transform'])
def test_callable_result_dtype_frame(keys, agg_index, input_dtype, result_dtype, method):
    df = DataFrame({'a': [1], 'b': [2], 'c': [True]})
    df['c'] = df['c'].astype(input_dtype)
    op = getattr(df.groupby(keys)[['c']], method)
    result = op(lambda x: x.astype(result_dtype).iloc[0])
    expected_index = pd.RangeIndex(0, 1) if method == 'transform' else agg_index
    expected = DataFrame({'c': [df['c'].iloc[0]]}, index=expected_index).astype(result_dtype)
    if method == 'apply':
        expected.columns.names = [0]
    tm.assert_frame_equal(result, expected)