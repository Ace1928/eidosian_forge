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
@pytest.mark.parametrize('date_string', ['32/32/2019', '02/30/2019', '13/13/2019', '13/2019', 'a3/11/2018', '10/11/2o17'])
def test_invalid_parse_delimited_date(all_parsers, date_string):
    parser = all_parsers
    expected = DataFrame({0: [date_string]}, dtype='object')
    result = parser.read_csv(StringIO(date_string), header=None, parse_dates=[0])
    tm.assert_frame_equal(result, expected)