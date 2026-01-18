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
@xfail_pyarrow
def test_date_parser_resolution_if_not_ns(all_parsers):
    parser = all_parsers
    data = 'date,time,prn,rxstatus\n2013-11-03,19:00:00,126,00E80000\n2013-11-03,19:00:00,23,00E80000\n2013-11-03,19:00:00,13,00E80000\n'

    def date_parser(dt, time):
        try:
            arr = dt + 'T' + time
        except TypeError:
            arr = [datetime.combine(d, t) for d, t in zip(dt, time)]
        return np.array(arr, dtype='datetime64[s]')
    result = parser.read_csv_check_warnings(FutureWarning, "use 'date_format' instead", StringIO(data), date_parser=date_parser, parse_dates={'datetime': ['date', 'time']}, index_col=['datetime', 'prn'])
    datetimes = np.array(['2013-11-03T19:00:00'] * 3, dtype='datetime64[s]')
    expected = DataFrame(data={'rxstatus': ['00E80000'] * 3}, index=MultiIndex.from_arrays([datetimes, [126, 23, 13]], names=['datetime', 'prn']))
    tm.assert_frame_equal(result, expected)