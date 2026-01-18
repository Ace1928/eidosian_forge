import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_melt_preserves_datetime(self):
    df = DataFrame(data=[{'type': 'A0', 'start_date': pd.Timestamp('2023/03/01', tz='Asia/Tokyo'), 'end_date': pd.Timestamp('2023/03/10', tz='Asia/Tokyo')}, {'type': 'A1', 'start_date': pd.Timestamp('2023/03/01', tz='Asia/Tokyo'), 'end_date': pd.Timestamp('2023/03/11', tz='Asia/Tokyo')}], index=['aaaa', 'bbbb'])
    result = df.melt(id_vars=['type'], value_vars=['start_date', 'end_date'], var_name='start/end', value_name='date')
    expected = DataFrame({'type': {0: 'A0', 1: 'A1', 2: 'A0', 3: 'A1'}, 'start/end': {0: 'start_date', 1: 'start_date', 2: 'end_date', 3: 'end_date'}, 'date': {0: pd.Timestamp('2023-03-01 00:00:00+0900', tz='Asia/Tokyo'), 1: pd.Timestamp('2023-03-01 00:00:00+0900', tz='Asia/Tokyo'), 2: pd.Timestamp('2023-03-10 00:00:00+0900', tz='Asia/Tokyo'), 3: pd.Timestamp('2023-03-11 00:00:00+0900', tz='Asia/Tokyo')}})
    tm.assert_frame_equal(result, expected)