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
@pytest.mark.parametrize('func, expected_data, result_dtype_dict', [('sum', [[2, 4], [10, 12], [18, 20]], {10: 'int64', 20: 'int64'}), ('std', [[2 ** 0.5] * 2] * 3, 'float64'), ('var', [[2] * 2] * 3, {10: 'float64', 20: 'float64'})])
def test_groupby_mixed_cols_axis1(func, expected_data, result_dtype_dict):
    df = DataFrame(np.arange(12).reshape(3, 4), index=Index([0, 1, 0], name='y'), columns=Index([10, 20, 10, 20], name='x'), dtype='int64').astype({10: 'Int64'})
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby('x', axis=1)
    result = gb.agg(func)
    expected = DataFrame(data=expected_data, index=Index([0, 1, 0], name='y'), columns=Index([10, 20], name='x')).astype(result_dtype_dict)
    tm.assert_frame_equal(result, expected)