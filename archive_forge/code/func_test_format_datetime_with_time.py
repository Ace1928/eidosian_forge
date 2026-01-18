from datetime import datetime
import dateutil.tz
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_format_datetime_with_time(self):
    dti = DatetimeIndex([datetime(2012, 2, 7), datetime(2012, 2, 7, 23)])
    msg = 'DatetimeIndex.format is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = dti.format()
    expected = ['2012-02-07 00:00:00', '2012-02-07 23:00:00']
    assert len(result) == 2
    assert result == expected