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
def test_date_parser_and_names(all_parsers):
    parser = all_parsers
    data = StringIO('x,y\n1,2')
    warn = UserWarning
    if parser.engine == 'pyarrow':
        warn = (UserWarning, DeprecationWarning)
    result = parser.read_csv_check_warnings(warn, 'Could not infer format', data, parse_dates=['B'], names=['B'])
    expected = DataFrame({'B': ['y', '2']}, index=['x', '1'])
    tm.assert_frame_equal(result, expected)