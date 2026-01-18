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
def test_date_parser_usecols_thousands(all_parsers):
    data = 'A,B,C\n    1,3,20-09-01-01\n    2,4,20-09-01-01\n    '
    parser = all_parsers
    if parser.engine == 'pyarrow':
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), parse_dates=[1], usecols=[1, 2], thousands='-')
        return
    result = parser.read_csv_check_warnings(UserWarning, 'Could not infer format', StringIO(data), parse_dates=[1], usecols=[1, 2], thousands='-')
    expected = DataFrame({'B': [3, 4], 'C': [Timestamp('20-09-2001 01:00:00')] * 2})
    tm.assert_frame_equal(result, expected)