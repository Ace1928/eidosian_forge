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
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_date_unit(self, unit, datetime_frame):
    df = datetime_frame
    df['date'] = Timestamp('20130101 20:43:42').as_unit('ns')
    dl = df.columns.get_loc('date')
    df.iloc[1, dl] = Timestamp('19710101 20:43:42')
    df.iloc[2, dl] = Timestamp('21460101 20:43:42')
    df.iloc[4, dl] = pd.NaT
    json = df.to_json(date_format='epoch', date_unit=unit)
    result = read_json(StringIO(json), date_unit=unit)
    tm.assert_frame_equal(result, df)
    result = read_json(StringIO(json), date_unit=None)
    tm.assert_frame_equal(result, df)