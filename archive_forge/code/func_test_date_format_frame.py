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
@pytest.mark.parametrize('date,date_unit', [('20130101 20:43:42.123', None), ('20130101 20:43:42', 's'), ('20130101 20:43:42.123', 'ms'), ('20130101 20:43:42.123456', 'us'), ('20130101 20:43:42.123456789', 'ns')])
def test_date_format_frame(self, date, date_unit, datetime_frame):
    df = datetime_frame
    df['date'] = Timestamp(date).as_unit('ns')
    df.iloc[1, df.columns.get_loc('date')] = pd.NaT
    df.iloc[5, df.columns.get_loc('date')] = pd.NaT
    if date_unit:
        json = df.to_json(date_format='iso', date_unit=date_unit)
    else:
        json = df.to_json(date_format='iso')
    result = read_json(StringIO(json))
    expected = df.copy()
    tm.assert_frame_equal(result, expected)