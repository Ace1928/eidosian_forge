from datetime import datetime
import dateutil.tz
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_format_date_explicit_date_format(self):
    dti = pd.to_datetime([datetime(2003, 2, 1), NaT])
    msg = 'DatetimeIndex.format is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        formatted = dti.format(date_format='%m-%d-%Y', na_rep='UT')
    assert formatted[0] == '02-01-2003'
    assert formatted[1] == 'UT'