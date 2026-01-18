import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isin_dupe_self(self):
    other = DataFrame({'A': [1, 0, 1, 0], 'B': [1, 1, 0, 0]})
    df = DataFrame([[1, 1], [1, 0], [0, 0]], columns=['A', 'A'])
    result = df.isin(other)
    expected = DataFrame(False, index=df.index, columns=df.columns)
    expected.loc[0] = True
    expected.iloc[1, 1] = True
    tm.assert_frame_equal(result, expected)