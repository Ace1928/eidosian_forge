import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('func', [min, max, np.min, np.max, 'first', 'last'])
def test_groupby_transform_timezone_column(func):
    ts = pd.to_datetime('now', utc=True).tz_convert('Asia/Singapore')
    result = DataFrame({'end_time': [ts], 'id': [1]})
    warn = FutureWarning if not isinstance(func, str) else None
    msg = 'using SeriesGroupBy.[min|max]'
    with tm.assert_produces_warning(warn, match=msg):
        result['max_end_time'] = result.groupby('id').end_time.transform(func)
    expected = DataFrame([[ts, 1, ts]], columns=['end_time', 'id', 'max_end_time'])
    tm.assert_frame_equal(result, expected)