import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_mi_add_cell_missing_row_non_unique():
    result = DataFrame([[1, 2, 5, 6], [3, 4, 7, 8]], index=['a', 'a'], columns=MultiIndex.from_product([[1, 2], ['A', 'B']]))
    result.loc['c'] = -1
    result.loc['c', (1, 'A')] = 3
    result.loc['d', (1, 'A')] = 3
    expected = DataFrame([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0], [3.0, -1.0, -1, -1], [3.0, np.nan, np.nan, np.nan]], index=['a', 'a', 'c', 'd'], columns=MultiIndex.from_product([[1, 2], ['A', 'B']]))
    tm.assert_frame_equal(result, expected)