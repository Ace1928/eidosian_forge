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
def test_dates_display(self):
    x = Series(date_range('20130101 09:00:00', periods=5, freq='D'))
    x.iloc[1] = np.nan
    result = fmt._Datetime64Formatter(x._values).get_result()
    assert result[0].strip() == '2013-01-01 09:00:00'
    assert result[1].strip() == 'NaT'
    assert result[4].strip() == '2013-01-05 09:00:00'
    x = Series(date_range('20130101 09:00:00', periods=5, freq='s'))
    x.iloc[1] = np.nan
    result = fmt._Datetime64Formatter(x._values).get_result()
    assert result[0].strip() == '2013-01-01 09:00:00'
    assert result[1].strip() == 'NaT'
    assert result[4].strip() == '2013-01-01 09:00:04'
    x = Series(date_range('20130101 09:00:00', periods=5, freq='ms'))
    x.iloc[1] = np.nan
    result = fmt._Datetime64Formatter(x._values).get_result()
    assert result[0].strip() == '2013-01-01 09:00:00.000'
    assert result[1].strip() == 'NaT'
    assert result[4].strip() == '2013-01-01 09:00:00.004'
    x = Series(date_range('20130101 09:00:00', periods=5, freq='us'))
    x.iloc[1] = np.nan
    result = fmt._Datetime64Formatter(x._values).get_result()
    assert result[0].strip() == '2013-01-01 09:00:00.000000'
    assert result[1].strip() == 'NaT'
    assert result[4].strip() == '2013-01-01 09:00:00.000004'
    x = Series(date_range('20130101 09:00:00', periods=5, freq='ns'))
    x.iloc[1] = np.nan
    result = fmt._Datetime64Formatter(x._values).get_result()
    assert result[0].strip() == '2013-01-01 09:00:00.000000000'
    assert result[1].strip() == 'NaT'
    assert result[4].strip() == '2013-01-01 09:00:00.000000004'