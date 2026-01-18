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
@pytest.mark.parametrize('unit', ['s', 'ms', 'us'])
def test_iso_non_nano_datetimes(self, unit):
    index = DatetimeIndex([np.datetime64('2023-01-01T11:22:33.123456', unit)], dtype=f'datetime64[{unit}]')
    df = DataFrame({'date': Series([np.datetime64('2022-01-01T11:22:33.123456', unit)], dtype=f'datetime64[{unit}]', index=index), 'date_obj': Series([np.datetime64('2023-01-01T11:22:33.123456', unit)], dtype=object, index=index)})
    buf = StringIO()
    df.to_json(buf, date_format='iso', date_unit=unit)
    buf.seek(0)
    tm.assert_frame_equal(read_json(buf, convert_dates=['date', 'date_obj']), df, check_index_type=False, check_dtype=False)