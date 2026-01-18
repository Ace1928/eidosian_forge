import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_inner_sort_columns(self):
    df1 = DataFrame({'A': [0], 'B': [1], 0: 1})
    df2 = DataFrame({'A': [100], 0: 2})
    result = concat([df1, df2], ignore_index=True, join='inner', sort=True)
    expected = DataFrame({0: [1, 2], 'A': [0, 100]})
    tm.assert_frame_equal(result, expected)