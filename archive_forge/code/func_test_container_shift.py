import inspect
import operator
import numpy as np
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort
@pytest.mark.parametrize('frame', [True, False])
@pytest.mark.parametrize('periods, indices', [(-2, [2, 3, 4, -1, -1]), (0, [0, 1, 2, 3, 4]), (2, [-1, -1, 0, 1, 2])])
def test_container_shift(self, data, frame, periods, indices):
    subset = data[:5]
    data = pd.Series(subset, name='A')
    expected = pd.Series(subset.take(indices, allow_fill=True), name='A')
    if frame:
        result = data.to_frame(name='A').assign(B=1).shift(periods)
        expected = pd.concat([expected, pd.Series([1] * 5, name='B').shift(periods)], axis=1)
        compare = tm.assert_frame_equal
    else:
        result = data.shift(periods)
        compare = tm.assert_series_equal
    compare(result, expected)