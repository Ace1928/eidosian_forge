from datetime import (
from io import StringIO
from dateutil.parser import parse as du_parse
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import parsing
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.tools.datetimes import start_caching_at
from pandas.io.parsers import read_csv
def test_dayfirst_warnings():
    input = 'date\n31/12/2014\n10/03/2011'
    expected = DatetimeIndex(['2014-12-31', '2011-03-10'], dtype='datetime64[ns]', freq=None, name='date')
    warning_msg = 'Parsing dates in .* format when dayfirst=.* was specified. Pass `dayfirst=.*` or specify a format to silence this warning.'
    res1 = read_csv(StringIO(input), parse_dates=['date'], dayfirst=True, index_col='date').index
    tm.assert_index_equal(expected, res1)
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res2 = read_csv(StringIO(input), parse_dates=['date'], dayfirst=False, index_col='date').index
    tm.assert_index_equal(expected, res2)
    input = 'date\n31/12/2014\n03/30/2011'
    expected = Index(['31/12/2014', '03/30/2011'], dtype='object', name='date')
    res5 = read_csv(StringIO(input), parse_dates=['date'], dayfirst=True, index_col='date').index
    tm.assert_index_equal(expected, res5)
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res6 = read_csv(StringIO(input), parse_dates=['date'], dayfirst=False, index_col='date').index
    tm.assert_index_equal(expected, res6)