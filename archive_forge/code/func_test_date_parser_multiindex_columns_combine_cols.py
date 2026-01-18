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
@pytest.mark.parametrize('parse_spec, col_name', [([[('a', '1'), ('b', '2')]], ('a_b', '1_2')), ({('foo', '1'): [('a', '1'), ('b', '2')]}, ('foo', '1'))])
def test_date_parser_multiindex_columns_combine_cols(all_parsers, parse_spec, col_name):
    parser = all_parsers
    data = 'a,b,c\n1,2,3\n2019-12,-31,6'
    depr_msg = "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), parse_dates=parse_spec, header=[0, 1])
    expected = DataFrame({col_name: Timestamp('2019-12-31').as_unit('ns'), ('c', '3'): [6]})
    tm.assert_frame_equal(result, expected)