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
def test_groupby_duplicate_index():
    ser = Series([2, 5, 6, 8], index=[2.0, 4.0, 4.0, 5.0])
    gb = ser.groupby(level=0)
    result = gb.mean()
    expected = Series([2, 5.5, 8], index=[2.0, 4.0, 5.0])
    tm.assert_series_equal(result, expected)