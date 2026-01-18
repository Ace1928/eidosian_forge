import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_axis_columns_ignore_index():
    df = DataFrame([[1, 2]], columns=['d', 'c'])
    result = df.sort_index(axis='columns', ignore_index=True)
    expected = DataFrame([[2, 1]])
    tm.assert_frame_equal(result, expected)