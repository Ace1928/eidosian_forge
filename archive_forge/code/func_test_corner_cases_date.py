from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
def test_corner_cases_date(simple_date_range_series, unit):
    ts = simple_date_range_series('2000-04-28', '2000-04-30 11:00', freq='h')
    ts.index = ts.index.as_unit(unit)
    msg = "The 'kind' keyword in Series.resample is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ts.resample('ME', kind='period').mean()
    assert len(result) == 1
    assert result.index[0] == Period('2000-04', freq='M')