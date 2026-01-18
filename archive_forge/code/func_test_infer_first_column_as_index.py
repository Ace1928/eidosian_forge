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
@skip_pyarrow
def test_infer_first_column_as_index(all_parsers):
    parser = all_parsers
    data = 'a,b,c\n1970-01-01,2,3,4'
    result = parser.read_csv(StringIO(data), parse_dates=['a'])
    expected = DataFrame({'a': '2', 'b': 3, 'c': 4}, index=['1970-01-01'])
    tm.assert_frame_equal(result, expected)