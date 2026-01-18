from datetime import datetime
import dateutil.tz
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dti_representation(self, unit):
    idxs = []
    idxs.append(DatetimeIndex([], freq='D'))
    idxs.append(DatetimeIndex(['2011-01-01'], freq='D'))
    idxs.append(DatetimeIndex(['2011-01-01', '2011-01-02'], freq='D'))
    idxs.append(DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'], freq='D'))
    idxs.append(DatetimeIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00'], freq='h', tz='Asia/Tokyo'))
    idxs.append(DatetimeIndex(['2011-01-01 09:00', '2011-01-01 10:00', NaT], tz='US/Eastern'))
    idxs.append(DatetimeIndex(['2011-01-01 09:00', '2011-01-01 10:00', NaT], tz='UTC'))
    exp = []
    exp.append("DatetimeIndex([], dtype='datetime64[ns]', freq='D')")
    exp.append("DatetimeIndex(['2011-01-01'], dtype='datetime64[ns]', freq='D')")
    exp.append("DatetimeIndex(['2011-01-01', '2011-01-02'], dtype='datetime64[ns]', freq='D')")
    exp.append("DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'], dtype='datetime64[ns]', freq='D')")
    exp.append("DatetimeIndex(['2011-01-01 09:00:00+09:00', '2011-01-01 10:00:00+09:00', '2011-01-01 11:00:00+09:00'], dtype='datetime64[ns, Asia/Tokyo]', freq='h')")
    exp.append("DatetimeIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00', 'NaT'], dtype='datetime64[ns, US/Eastern]', freq=None)")
    exp.append("DatetimeIndex(['2011-01-01 09:00:00+00:00', '2011-01-01 10:00:00+00:00', 'NaT'], dtype='datetime64[ns, UTC]', freq=None)")
    with pd.option_context('display.width', 300):
        for index, expected in zip(idxs, exp):
            index = index.as_unit(unit)
            expected = expected.replace('[ns', f'[{unit}')
            result = repr(index)
            assert result == expected
            result = str(index)
            assert result == expected