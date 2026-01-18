from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_datetime64(self):
    idx = DatetimeIndex(['2016-05-16', 'NaT', NaT, np.nan], dtype='M8[ns]', name='idx')
    result = idx.astype('datetime64[ns]')
    tm.assert_index_equal(result, idx)
    assert result is not idx
    result = idx.astype('datetime64[ns]', copy=False)
    tm.assert_index_equal(result, idx)
    assert result is idx
    idx_tz = DatetimeIndex(['2016-05-16', 'NaT', NaT, np.nan], tz='EST', name='idx')
    msg = 'Cannot use .astype to convert from timezone-aware'
    with pytest.raises(TypeError, match=msg):
        result = idx_tz.astype('datetime64[ns]')