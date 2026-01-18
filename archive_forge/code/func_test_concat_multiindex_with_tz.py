import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_multiindex_with_tz(self):
    df = DataFrame({'dt': DatetimeIndex([datetime(2014, 1, 1), datetime(2014, 1, 2), datetime(2014, 1, 3)], dtype='M8[ns, US/Pacific]'), 'b': ['A', 'B', 'C'], 'c': [1, 2, 3], 'd': [4, 5, 6]})
    df = df.set_index(['dt', 'b'])
    exp_idx1 = DatetimeIndex(['2014-01-01', '2014-01-02', '2014-01-03'] * 2, dtype='M8[ns, US/Pacific]', name='dt')
    exp_idx2 = Index(['A', 'B', 'C'] * 2, name='b')
    exp_idx = MultiIndex.from_arrays([exp_idx1, exp_idx2])
    expected = DataFrame({'c': [1, 2, 3] * 2, 'd': [4, 5, 6] * 2}, index=exp_idx, columns=['c', 'd'])
    result = concat([df, df])
    tm.assert_frame_equal(result, expected)