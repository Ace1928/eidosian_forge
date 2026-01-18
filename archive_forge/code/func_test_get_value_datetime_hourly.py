from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq', ['h', 'D'])
def test_get_value_datetime_hourly(self, freq):
    dti = date_range('2016-01-01', periods=3, freq='MS')
    pi = dti.to_period(freq)
    ser = Series(range(7, 10), index=pi)
    ts = dti[0]
    assert pi.get_loc(ts) == 0
    assert ser[ts] == 7
    assert ser.loc[ts] == 7
    ts2 = ts + Timedelta(hours=3)
    if freq == 'h':
        with pytest.raises(KeyError, match='2016-01-01 03:00'):
            pi.get_loc(ts2)
        with pytest.raises(KeyError, match='2016-01-01 03:00'):
            ser[ts2]
        with pytest.raises(KeyError, match='2016-01-01 03:00'):
            ser.loc[ts2]
    else:
        assert pi.get_loc(ts2) == 0
        assert ser[ts2] == 7
        assert ser.loc[ts2] == 7