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
def test_convert_dates(self, datetime_series, datetime_frame):
    df = datetime_frame
    df['date'] = Timestamp('20130101').as_unit('ns')
    json = StringIO(df.to_json())
    result = read_json(json)
    tm.assert_frame_equal(result, df)
    df['foo'] = 1.0
    json = StringIO(df.to_json(date_unit='ns'))
    result = read_json(json, convert_dates=False)
    expected = df.copy()
    expected['date'] = expected['date'].values.view('i8')
    expected['foo'] = expected['foo'].astype('int64')
    tm.assert_frame_equal(result, expected)
    ts = Series(Timestamp('20130101').as_unit('ns'), index=datetime_series.index)
    json = StringIO(ts.to_json())
    result = read_json(json, typ='series')
    tm.assert_series_equal(result, ts)