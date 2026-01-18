import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data, expected_first, expected_last', [({'id': ['A'], 'time': Timestamp('2012-02-01 14:00:00', tz='US/Central'), 'foo': [1]}, {'id': ['A'], 'time': Timestamp('2012-02-01 14:00:00', tz='US/Central'), 'foo': [1]}, {'id': ['A'], 'time': Timestamp('2012-02-01 14:00:00', tz='US/Central'), 'foo': [1]}), ({'id': ['A', 'B', 'A'], 'time': [Timestamp('2012-01-01 13:00:00', tz='America/New_York'), Timestamp('2012-02-01 14:00:00', tz='US/Central'), Timestamp('2012-03-01 12:00:00', tz='Europe/London')], 'foo': [1, 2, 3]}, {'id': ['A', 'B'], 'time': [Timestamp('2012-01-01 13:00:00', tz='America/New_York'), Timestamp('2012-02-01 14:00:00', tz='US/Central')], 'foo': [1, 2]}, {'id': ['A', 'B'], 'time': [Timestamp('2012-03-01 12:00:00', tz='Europe/London'), Timestamp('2012-02-01 14:00:00', tz='US/Central')], 'foo': [3, 2]})])
def test_first_last_tz(data, expected_first, expected_last):
    df = DataFrame(data)
    result = df.groupby('id', as_index=False).first()
    expected = DataFrame(expected_first)
    cols = ['id', 'time', 'foo']
    tm.assert_frame_equal(result[cols], expected[cols])
    result = df.groupby('id', as_index=False)['time'].first()
    tm.assert_frame_equal(result, expected[['id', 'time']])
    result = df.groupby('id', as_index=False).last()
    expected = DataFrame(expected_last)
    cols = ['id', 'time', 'foo']
    tm.assert_frame_equal(result[cols], expected[cols])
    result = df.groupby('id', as_index=False)['time'].last()
    tm.assert_frame_equal(result, expected[['id', 'time']])