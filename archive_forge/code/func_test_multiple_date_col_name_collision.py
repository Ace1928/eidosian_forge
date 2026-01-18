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
@pytest.mark.parametrize('data,parse_dates,msg', [('date_NominalTime,date,NominalTime\nKORD1,19990127, 19:00:00\nKORD2,19990127, 20:00:00', [[1, 2]], 'New date column already in dict date_NominalTime'), ('ID,date,nominalTime\nKORD,19990127, 19:00:00\nKORD,19990127, 20:00:00', {'ID': [1, 2]}, 'Date column ID already in dict')])
def test_multiple_date_col_name_collision(all_parsers, data, parse_dates, msg):
    parser = all_parsers
    depr_msg = "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning((FutureWarning, DeprecationWarning), match=depr_msg, check_stacklevel=False):
            parser.read_csv(StringIO(data), parse_dates=parse_dates)