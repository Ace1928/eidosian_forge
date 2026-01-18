from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('start', ['2018-12-02 21:50:00+00:00', Timestamp('2018-12-02 21:50:00+00:00'), Timestamp('2018-12-02 21:50:00+00:00').to_pydatetime()])
@pytest.mark.parametrize('end', ['2018-12-02 21:52:00+00:00', Timestamp('2018-12-02 21:52:00+00:00'), Timestamp('2018-12-02 21:52:00+00:00').to_pydatetime()])
def test_getitem_with_datestring_with_UTC_offset(self, start, end):
    idx = date_range(start='2018-12-02 14:50:00-07:00', end='2018-12-02 14:50:00-07:00', freq='1min')
    df = DataFrame(1, index=idx, columns=['A'])
    result = df[start:end]
    expected = df.iloc[0:3, :]
    tm.assert_frame_equal(result, expected)
    start = str(start)
    end = str(end)
    with pytest.raises(ValueError, match='Both dates must'):
        df[start:end[:-4] + '1:00']
    with pytest.raises(ValueError, match='The index must be timezone'):
        df = df.tz_localize(None)
        df[start:end]