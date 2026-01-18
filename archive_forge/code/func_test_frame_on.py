import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_frame_on(self):
    df = DataFrame({'B': range(5), 'C': date_range('20130101 09:00:00', periods=5, freq='3s')})
    df['A'] = [Timestamp('20130101 09:00:00'), Timestamp('20130101 09:00:02'), Timestamp('20130101 09:00:03'), Timestamp('20130101 09:00:05'), Timestamp('20130101 09:00:06')]
    expected = df.set_index('A').rolling('2s').B.sum().reset_index(drop=True)
    result = df.rolling('2s', on='A').B.sum()
    tm.assert_series_equal(result, expected)
    expected = df.set_index('A').rolling('2s')[['B']].sum().reset_index()[['B', 'A']]
    result = df.rolling('2s', on='A')[['B']].sum()
    tm.assert_frame_equal(result, expected)