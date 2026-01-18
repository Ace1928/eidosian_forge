import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('func', ['ffill', 'bfill', 'shift'])
@pytest.mark.parametrize('key, val', [('level', 0), ('by', Series([0]))])
def test_ffill_not_in_axis(func, key, val):
    df = DataFrame([[np.nan]])
    result = getattr(df.groupby(**{key: val}), func)()
    expected = df
    tm.assert_frame_equal(result, expected)