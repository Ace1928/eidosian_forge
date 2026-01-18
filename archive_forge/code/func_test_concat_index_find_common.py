from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['Int8', 'Int16', 'Int32'])
def test_concat_index_find_common(self, dtype):
    df1 = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype=dtype))
    df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype='Int32'))
    result = concat([df1, df2], ignore_index=True, join='outer', sort=True)
    expected = DataFrame([[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype='Int32'))
    tm.assert_frame_equal(result, expected)