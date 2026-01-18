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
@pytest.mark.parametrize('left_empty', [True, False])
@pytest.mark.parametrize('right_empty', [True, False])
def test_merge_empty_frames_column_order(left_empty, right_empty):
    df1 = DataFrame(1, index=[0], columns=['A', 'B'])
    df2 = DataFrame(1, index=[0], columns=['A', 'C', 'D'])
    if left_empty:
        df1 = df1.iloc[:0]
    if right_empty:
        df2 = df2.iloc[:0]
    result = merge(df1, df2, on=['A'], how='outer')
    expected = DataFrame(1, index=[0], columns=['A', 'B', 'C', 'D'])
    if left_empty and right_empty:
        expected = expected.iloc[:0]
    elif left_empty:
        expected['B'] = np.nan
    elif right_empty:
        expected[['C', 'D']] = np.nan
    tm.assert_frame_equal(result, expected)