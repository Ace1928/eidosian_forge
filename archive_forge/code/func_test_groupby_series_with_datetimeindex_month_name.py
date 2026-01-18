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
def test_groupby_series_with_datetimeindex_month_name():
    s = Series([0, 1, 0], index=date_range('2022-01-01', periods=3), name='jan')
    result = s.groupby(s).count()
    expected = Series([2, 1], name='jan')
    expected.index.name = 'jan'
    tm.assert_series_equal(result, expected)