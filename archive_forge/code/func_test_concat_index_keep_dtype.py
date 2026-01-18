from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_index_keep_dtype(self):
    df1 = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype='object'))
    df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype='object'))
    result = concat([df1, df2], ignore_index=True, join='outer', sort=True)
    expected = DataFrame([[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype='object'))
    tm.assert_frame_equal(result, expected)