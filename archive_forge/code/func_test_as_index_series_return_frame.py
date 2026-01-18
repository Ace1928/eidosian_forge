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
def test_as_index_series_return_frame(df):
    grouped = df.groupby('A', as_index=False)
    grouped2 = df.groupby(['A', 'B'], as_index=False)
    result = grouped['C'].agg('sum')
    expected = grouped.agg('sum').loc[:, ['A', 'C']]
    assert isinstance(result, DataFrame)
    tm.assert_frame_equal(result, expected)
    result2 = grouped2['C'].agg('sum')
    expected2 = grouped2.agg('sum').loc[:, ['A', 'B', 'C']]
    assert isinstance(result2, DataFrame)
    tm.assert_frame_equal(result2, expected2)
    result = grouped['C'].sum()
    expected = grouped.sum().loc[:, ['A', 'C']]
    assert isinstance(result, DataFrame)
    tm.assert_frame_equal(result, expected)
    result2 = grouped2['C'].sum()
    expected2 = grouped2.sum().loc[:, ['A', 'B', 'C']]
    assert isinstance(result2, DataFrame)
    tm.assert_frame_equal(result2, expected2)