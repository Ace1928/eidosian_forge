from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
def test_datetimelike_frame(self):
    df = DataFrame({'date': [Timestamp('20130101').tz_localize('UTC')] + [NaT] * 5})
    with option_context('display.max_rows', 5):
        result = str(df)
        assert '2013-01-01 00:00:00+00:00' in result
        assert 'NaT' in result
        assert '...' in result
        assert '[6 rows x 1 columns]' in result
    dts = [Timestamp('2011-01-01', tz='US/Eastern')] * 5 + [NaT] * 5
    df = DataFrame({'dt': dts, 'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    with option_context('display.max_rows', 5):
        expected = '                          dt   x\n0  2011-01-01 00:00:00-05:00   1\n1  2011-01-01 00:00:00-05:00   2\n..                       ...  ..\n8                        NaT   9\n9                        NaT  10\n\n[10 rows x 2 columns]'
        assert repr(df) == expected
    dts = [NaT] * 5 + [Timestamp('2011-01-01', tz='US/Eastern')] * 5
    df = DataFrame({'dt': dts, 'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    with option_context('display.max_rows', 5):
        expected = '                          dt   x\n0                        NaT   1\n1                        NaT   2\n..                       ...  ..\n8  2011-01-01 00:00:00-05:00   9\n9  2011-01-01 00:00:00-05:00  10\n\n[10 rows x 2 columns]'
        assert repr(df) == expected
    dts = [Timestamp('2011-01-01', tz='Asia/Tokyo')] * 5 + [Timestamp('2011-01-01', tz='US/Eastern')] * 5
    df = DataFrame({'dt': dts, 'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    with option_context('display.max_rows', 5):
        expected = '                           dt   x\n0   2011-01-01 00:00:00+09:00   1\n1   2011-01-01 00:00:00+09:00   2\n..                        ...  ..\n8   2011-01-01 00:00:00-05:00   9\n9   2011-01-01 00:00:00-05:00  10\n\n[10 rows x 2 columns]'
        assert repr(df) == expected