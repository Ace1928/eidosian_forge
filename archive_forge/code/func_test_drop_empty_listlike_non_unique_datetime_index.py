import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('empty_listlike', [[], {}, np.array([]), Series([], dtype='datetime64[ns]'), Index([]), DatetimeIndex([])])
def test_drop_empty_listlike_non_unique_datetime_index(self, empty_listlike):
    data = {'column_a': [5, 10], 'column_b': ['one', 'two']}
    index = [Timestamp('2021-01-01'), Timestamp('2021-01-01')]
    df = DataFrame(data, index=index)
    expected = df.copy()
    result = df.drop(empty_listlike)
    tm.assert_frame_equal(result, expected)