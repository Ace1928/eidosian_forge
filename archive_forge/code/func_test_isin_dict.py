import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isin_dict(self):
    df = DataFrame({'A': ['a', 'b', 'c'], 'B': ['a', 'e', 'f']})
    d = {'A': ['a']}
    expected = DataFrame(False, df.index, df.columns)
    expected.loc[0, 'A'] = True
    result = df.isin(d)
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'A': ['a', 'b', 'c'], 'B': ['a', 'e', 'f']})
    df.columns = ['A', 'A']
    expected = DataFrame(False, df.index, df.columns)
    expected.loc[0, 'A'] = True
    result = df.isin(d)
    tm.assert_frame_equal(result, expected)