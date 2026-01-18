import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_index_differently_ordered_slice_none():
    df = DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]], columns=['a', 'b'])
    result = df.loc[(slice(None), [2, 1]), :]
    expected = DataFrame([[3, 4], [7, 8], [1, 2], [5, 6]], index=[['a', 'b', 'a', 'b'], [2, 2, 1, 1]], columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)