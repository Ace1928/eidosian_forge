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
@pytest.mark.parametrize('cache_dates', [True, False])
@pytest.mark.parametrize('value', ['0'])
def test_bad_date_parse_with_warning(all_parsers, cache_dates, value):
    parser = all_parsers
    s = StringIO(f'{value},\n' * 50000)
    if parser.engine == 'pyarrow':
        warn = None
    elif cache_dates:
        warn = None
    else:
        warn = UserWarning
    parser.read_csv_check_warnings(warn, 'Could not infer format', s, header=None, names=['foo', 'bar'], parse_dates=['foo'], cache_dates=cache_dates, raise_on_extra_warnings=False)