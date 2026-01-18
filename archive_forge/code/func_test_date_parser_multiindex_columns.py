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
def test_date_parser_multiindex_columns(all_parsers):
    parser = all_parsers
    data = 'a,b\n1,2\n2019-12-31,6'
    result = parser.read_csv(StringIO(data), parse_dates=[('a', '1')], header=[0, 1])
    expected = DataFrame({('a', '1'): Timestamp('2019-12-31').as_unit('ns'), ('b', '2'): [6]})
    tm.assert_frame_equal(result, expected)