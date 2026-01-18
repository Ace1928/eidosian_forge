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
@pytest.mark.parametrize('test_series', [True, False])
@pytest.mark.parametrize('kwarg, value, name, warn', [('by', 'a', 1, None), ('by', ['a'], 1, FutureWarning), ('by', ['a'], (1,), None), ('level', 0, 1, None), ('level', [0], 1, FutureWarning), ('level', [0], (1,), None)])
def test_depr_get_group_len_1_list_likes(test_series, kwarg, value, name, warn):
    obj = DataFrame({'b': [3, 4, 5]}, index=Index([1, 1, 2], name='a'))
    if test_series:
        obj = obj['b']
    gb = obj.groupby(**{kwarg: value})
    msg = 'you will need to pass a length-1 tuple'
    with tm.assert_produces_warning(warn, match=msg):
        result = gb.get_group(name)
    if test_series:
        expected = Series([3, 4], index=Index([1, 1], name='a'), name='b')
    else:
        expected = DataFrame({'b': [3, 4]}, index=Index([1, 1], name='a'))
    tm.assert_equal(result, expected)