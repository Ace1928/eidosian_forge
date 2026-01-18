from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_multi_level_stack_categorical(self, future_stack):
    midx = MultiIndex.from_arrays([['A'] * 2 + ['B'] * 2, pd.Categorical(list('abab')), pd.Categorical(list('ccdd'))])
    df = DataFrame(np.arange(8).reshape(2, 4), columns=midx)
    result = df.stack([1, 2], future_stack=future_stack)
    if future_stack:
        expected = DataFrame([[0, np.nan], [1, np.nan], [np.nan, 2], [np.nan, 3], [4, np.nan], [5, np.nan], [np.nan, 6], [np.nan, 7]], columns=['A', 'B'], index=MultiIndex.from_arrays([[0] * 4 + [1] * 4, pd.Categorical(list('abababab')), pd.Categorical(list('ccddccdd'))]))
    else:
        expected = DataFrame([[0, np.nan], [np.nan, 2], [1, np.nan], [np.nan, 3], [4, np.nan], [np.nan, 6], [5, np.nan], [np.nan, 7]], columns=['A', 'B'], index=MultiIndex.from_arrays([[0] * 4 + [1] * 4, pd.Categorical(list('aabbaabb')), pd.Categorical(list('cdcdcdcd'))]))
    tm.assert_frame_equal(result, expected)