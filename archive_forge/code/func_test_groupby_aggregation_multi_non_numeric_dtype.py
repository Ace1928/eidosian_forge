from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_groupby_aggregation_multi_non_numeric_dtype():
    df = DataFrame({'x': [1, 0, 1, 1, 0], 'y': [Timedelta(i, 'days') for i in range(1, 6)], 'z': [Timedelta(i * 10, 'days') for i in range(1, 6)]})
    expected = DataFrame({'y': [Timedelta(i, 'days') for i in range(7, 9)], 'z': [Timedelta(i * 10, 'days') for i in range(7, 9)]}, index=Index([0, 1], dtype='int64', name='x'))
    gb = df.groupby(by=['x'])
    result = gb.sum()
    tm.assert_frame_equal(result, expected)