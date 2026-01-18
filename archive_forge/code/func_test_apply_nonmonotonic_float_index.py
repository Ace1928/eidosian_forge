from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('arg,idx', [[[1, 2, 3], [0.1, 0.3, 0.2]], [[1, 2, 3], [0.1, 0.2, 0.3]], [[1, 4, 3], [0.1, 0.4, 0.2]]])
def test_apply_nonmonotonic_float_index(arg, idx):
    expected = DataFrame({'col': arg}, index=idx)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = expected.groupby('col', group_keys=False).apply(lambda x: x)
    tm.assert_frame_equal(result, expected)