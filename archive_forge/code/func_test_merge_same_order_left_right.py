from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize(('sort', 'values'), [(False, [1, 1, 0, 1, 1]), (True, [0, 1, 1, 1, 1])])
@pytest.mark.parametrize('how', ['left', 'right'])
def test_merge_same_order_left_right(self, sort, values, how):
    df = DataFrame({'a': [1, 0, 1]})
    result = df.merge(df, on='a', how=how, sort=sort)
    expected = DataFrame(values, columns=['a'])
    tm.assert_frame_equal(result, expected)