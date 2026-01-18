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
def test_multiple_date_col_timestamp_parse(all_parsers):
    parser = all_parsers
    data = '05/31/2012,15:30:00.029,1306.25,1,E,0,,1306.25\n05/31/2012,15:30:00.029,1306.25,8,E,0,,1306.25'
    result = parser.read_csv_check_warnings(FutureWarning, "use 'date_format' instead", StringIO(data), parse_dates=[[0, 1]], header=None, date_parser=Timestamp, raise_on_extra_warnings=False)
    expected = DataFrame([[Timestamp('05/31/2012, 15:30:00.029'), 1306.25, 1, 'E', 0, np.nan, 1306.25], [Timestamp('05/31/2012, 15:30:00.029'), 1306.25, 8, 'E', 0, np.nan, 1306.25]], columns=['0_1', 2, 3, 4, 5, 6, 7])
    tm.assert_frame_equal(result, expected)