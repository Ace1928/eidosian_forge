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
@pytest.mark.parametrize('names, usecols, parse_dates, missing_cols', [(None, ['val'], ['date', 'time'], 'date, time'), (None, ['val'], [0, 'time'], 'time'), (None, ['val'], [['date', 'time']], 'date, time'), (None, ['val'], [[0, 'time']], 'time'), (None, ['val'], {'date': [0, 'time']}, 'time'), (None, ['val'], {'date': ['date', 'time']}, 'date, time'), (None, ['val'], [['date', 'time'], 'date'], 'date, time'), (['date1', 'time1', 'temperature'], None, ['date', 'time'], 'date, time'), (['date1', 'time1', 'temperature'], ['date1', 'temperature'], ['date1', 'time'], 'time')])
def test_missing_parse_dates_column_raises(all_parsers, names, usecols, parse_dates, missing_cols):
    parser = all_parsers
    content = StringIO('date,time,val\n2020-01-31,04:20:32,32\n')
    msg = f"Missing column provided to 'parse_dates': '{missing_cols}'"
    depr_msg = "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    warn = FutureWarning
    if isinstance(parse_dates, list) and all((isinstance(x, (int, str)) for x in parse_dates)):
        warn = None
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(warn, match=depr_msg, check_stacklevel=False):
            parser.read_csv(content, sep=',', names=names, usecols=usecols, parse_dates=parse_dates)