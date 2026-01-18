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
@pytest.mark.parametrize('numeric_only', [True, False])
def test_groupby_empty_multi_column(as_index, numeric_only):
    df = DataFrame(data=[], columns=['A', 'B', 'C'])
    gb = df.groupby(['A', 'B'], as_index=as_index)
    result = gb.sum(numeric_only=numeric_only)
    if as_index:
        index = MultiIndex([[], []], [[], []], names=['A', 'B'])
        columns = ['C'] if not numeric_only else []
    else:
        index = RangeIndex(0)
        columns = ['A', 'B', 'C'] if not numeric_only else ['A', 'B']
    expected = DataFrame([], columns=columns, index=index)
    tm.assert_frame_equal(result, expected)