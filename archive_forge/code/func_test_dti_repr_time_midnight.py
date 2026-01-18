from datetime import datetime
import dateutil.tz
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dates, freq, expected_repr', [(['2012-01-01 00:00:00'], '60min', "DatetimeIndex(['2012-01-01 00:00:00'], dtype='datetime64[ns]', freq='60min')"), (['2012-01-01 00:00:00', '2012-01-01 01:00:00'], '60min', "DatetimeIndex(['2012-01-01 00:00:00', '2012-01-01 01:00:00'], dtype='datetime64[ns]', freq='60min')"), (['2012-01-01'], '24h', "DatetimeIndex(['2012-01-01'], dtype='datetime64[ns]', freq='24h')")])
def test_dti_repr_time_midnight(self, dates, freq, expected_repr, unit):
    dti = DatetimeIndex(dates, freq).as_unit(unit)
    actual_repr = repr(dti)
    assert actual_repr == expected_repr.replace('[ns]', f'[{unit}]')