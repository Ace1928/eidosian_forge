import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
def test_datetime_tz(self):
    tz_range = date_range('20130101', periods=3, tz='US/Eastern')
    tz_naive = tz_range.tz_convert('utc').tz_localize(None)
    df = DataFrame({'A': tz_range, 'B': date_range('20130101', periods=3)})
    df_naive = df.copy()
    df_naive['A'] = tz_naive
    expected = df_naive.to_json()
    assert expected == df.to_json()
    stz = Series(tz_range)
    s_naive = Series(tz_naive)
    assert stz.to_json() == s_naive.to_json()