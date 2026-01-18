from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_is_unchanged_when_other_methods_are_called_first(reduction_func):
    df = DataFrame({'a': [99, 99, 99, 88, 88, 88], 'b': [1, 2, 3, 4, 5, 6], 'c': [10, 20, 30, 40, 50, 60]})
    expected = DataFrame({'b': [15, 6], 'c': [150, 60]}, index=Index([88, 99], name='a'))
    grp = df.groupby(by='a')
    msg = 'The behavior of DataFrame.sum with axis=None is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False):
        result = grp.apply(sum, include_groups=False)
    tm.assert_frame_equal(result, expected)
    grp = df.groupby(by='a')
    args = get_groupby_method_args(reduction_func, df)
    _ = getattr(grp, reduction_func)(*args)
    with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False):
        result = grp.apply(sum, include_groups=False)
    tm.assert_frame_equal(result, expected)