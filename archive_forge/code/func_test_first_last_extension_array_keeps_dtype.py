import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('values', [pd.array([True, False], dtype='boolean'), pd.array([1, 2], dtype='Int64'), pd.to_datetime(['2020-01-01', '2020-02-01']), pd.to_timedelta([1, 2], unit='D')])
@pytest.mark.parametrize('function', ['first', 'last', 'min', 'max'])
def test_first_last_extension_array_keeps_dtype(values, function):
    df = DataFrame({'a': [1, 2], 'b': values})
    grouped = df.groupby('a')
    idx = Index([1, 2], name='a')
    expected_series = Series(values, name='b', index=idx)
    expected_frame = DataFrame({'b': values}, index=idx)
    result_series = getattr(grouped['b'], function)()
    tm.assert_series_equal(result_series, expected_series)
    result_frame = grouped.agg({'b': function})
    tm.assert_frame_equal(result_frame, expected_frame)