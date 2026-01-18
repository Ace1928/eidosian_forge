import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('func, expected, dtype, result_dtype_dict', [('sum', [5, 7, 9], 'int64', {}), ('std', [4.5 ** 0.5] * 3, int, {'i': float, 'j': float, 'k': float}), ('var', [4.5] * 3, int, {'i': float, 'j': float, 'k': float}), ('sum', [5, 7, 9], 'Int64', {'j': 'int64'}), ('std', [4.5 ** 0.5] * 3, 'Int64', {'i': float, 'j': float, 'k': float}), ('var', [4.5] * 3, 'Int64', {'i': 'float64', 'j': 'float64', 'k': 'float64'})])
def test_multiindex_groupby_mixed_cols_axis1(func, expected, dtype, result_dtype_dict):
    df = DataFrame([[1, 2, 3, 4, 5, 6]] * 3, columns=MultiIndex.from_product([['a', 'b'], ['i', 'j', 'k']])).astype({('a', 'j'): dtype, ('b', 'j'): dtype})
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(level=1, axis=1)
    result = gb.agg(func)
    expected = DataFrame([expected] * 3, columns=['i', 'j', 'k']).astype(result_dtype_dict)
    tm.assert_frame_equal(result, expected)