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
def test_nat_parse(all_parsers):
    parser = all_parsers
    df = DataFrame({'A': np.arange(10, dtype='float64'), 'B': Timestamp('20010101').as_unit('ns')})
    df.iloc[3:6, :] = np.nan
    with tm.ensure_clean('__nat_parse_.csv') as path:
        df.to_csv(path)
        result = parser.read_csv(path, index_col=0, parse_dates=['B'])
        tm.assert_frame_equal(result, df)