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
@pytest.mark.parametrize('sort_column', ['ints', 'floats', 'strings', ['ints', 'floats'], ['ints', 'strings']])
@pytest.mark.parametrize('group_column', ['int_groups', 'string_groups', ['int_groups', 'string_groups']])
def test_groupby_preserves_sort(sort_column, group_column):
    df = DataFrame({'int_groups': [3, 1, 0, 1, 0, 3, 3, 3], 'string_groups': ['z', 'a', 'z', 'a', 'a', 'g', 'g', 'g'], 'ints': [8, 7, 4, 5, 2, 9, 1, 1], 'floats': [2.3, 5.3, 6.2, -2.4, 2.2, 1.1, 1.1, 5], 'strings': ['z', 'd', 'a', 'e', 'word', 'word2', '42', '47']})
    df = df.sort_values(by=sort_column)
    g = df.groupby(group_column)

    def test_sort(x):
        tm.assert_frame_equal(x, x.sort_values(by=sort_column))
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        g.apply(test_sort)