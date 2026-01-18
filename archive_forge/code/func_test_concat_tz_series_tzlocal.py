import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_tz_series_tzlocal(self):
    x = [Timestamp('2011-01-01', tz=dateutil.tz.tzlocal()), Timestamp('2011-02-01', tz=dateutil.tz.tzlocal())]
    y = [Timestamp('2012-01-01', tz=dateutil.tz.tzlocal()), Timestamp('2012-02-01', tz=dateutil.tz.tzlocal())]
    result = concat([Series(x), Series(y)], ignore_index=True)
    tm.assert_series_equal(result, Series(x + y))
    assert result.dtype == 'datetime64[ns, tzlocal()]'