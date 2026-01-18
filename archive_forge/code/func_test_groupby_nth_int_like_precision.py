import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', [(Timestamp('2011-01-15 12:50:28.502376'), Timestamp('2011-01-20 12:50:28.593448')), (24650000000000001, 24650000000000002)])
def test_groupby_nth_int_like_precision(data):
    df = DataFrame({'a': [1, 1], 'b': data})
    grouped = df.groupby('a')
    result = grouped.nth(0)
    expected = DataFrame({'a': 1, 'b': [data[0]]})
    tm.assert_frame_equal(result, expected)