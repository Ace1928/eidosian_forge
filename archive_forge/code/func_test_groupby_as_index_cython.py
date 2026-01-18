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
def test_groupby_as_index_cython(df):
    data = df
    grouped = data.groupby('A', as_index=False)
    result = grouped.mean(numeric_only=True)
    expected = data.groupby(['A']).mean(numeric_only=True)
    expected.insert(0, 'A', expected.index)
    expected.index = RangeIndex(len(expected))
    tm.assert_frame_equal(result, expected)
    grouped = data.groupby(['A', 'B'], as_index=False)
    result = grouped.mean()
    expected = data.groupby(['A', 'B']).mean()
    arrays = list(zip(*expected.index.values))
    expected.insert(0, 'A', arrays[0])
    expected.insert(1, 'B', arrays[1])
    expected.index = RangeIndex(len(expected))
    tm.assert_frame_equal(result, expected)