import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_margin_dropna2(self):
    df = DataFrame({'a': [1, np.nan, np.nan, np.nan, 2, np.nan], 'b': [3, np.nan, 4, 4, 4, 4]})
    actual = crosstab(df.a, df.b, margins=True, dropna=True)
    expected = DataFrame([[1, 0, 1], [0, 1, 1], [1, 1, 2]])
    expected.index = Index([1.0, 2.0, 'All'], name='a')
    expected.columns = Index([3.0, 4.0, 'All'], name='b')
    tm.assert_frame_equal(actual, expected)