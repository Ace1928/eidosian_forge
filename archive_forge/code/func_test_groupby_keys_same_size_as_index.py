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
def test_groupby_keys_same_size_as_index():
    freq = 's'
    index = date_range(start=Timestamp('2015-09-29T11:34:44-0700'), periods=2, freq=freq)
    df = DataFrame([['A', 10], ['B', 15]], columns=['metric', 'values'], index=index)
    result = df.groupby([Grouper(level=0, freq=freq), 'metric']).mean()
    expected = df.set_index([df.index, 'metric']).astype(float)
    tm.assert_frame_equal(result, expected)