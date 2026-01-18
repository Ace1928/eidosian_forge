import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_mixed_non_unique_index(self):
    df1 = DataFrame({'a': [1, 2, 3, 4]}, index=[1, 2, 3, 'a'])
    df2 = DataFrame({'b': [5, 6, 7, 8]}, index=[1, 3, 3, 4])
    result = df1.join(df2)
    expected = DataFrame({'a': [1, 2, 3, 3, 4], 'b': [5, np.nan, 6, 7, np.nan]}, index=[1, 2, 3, 3, 'a'])
    tm.assert_frame_equal(result, expected)
    df3 = DataFrame({'a': [1, 2, 3, 4]}, index=[1, 2, 2, 'a'])
    df4 = DataFrame({'b': [5, 6, 7, 8]}, index=[1, 2, 3, 4])
    result = df3.join(df4)
    expected = DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 6, np.nan]}, index=[1, 2, 2, 'a'])
    tm.assert_frame_equal(result, expected)