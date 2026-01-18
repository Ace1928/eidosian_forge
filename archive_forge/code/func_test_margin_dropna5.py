import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_margin_dropna5(self):
    df = DataFrame({'a': [1, np.nan, np.nan, np.nan, 2, np.nan], 'b': [3, np.nan, 4, 4, 4, 4]})
    actual = crosstab(df.a, df.b, margins=True, dropna=False)
    expected = DataFrame([[1, 0, 0, 1.0], [0, 1, 0, 1.0], [0, 3, 1, np.nan], [1, 4, 0, 6.0]])
    expected.index = Index([1.0, 2.0, np.nan, 'All'], name='a')
    expected.columns = Index([3.0, 4.0, np.nan, 'All'], name='b')
    tm.assert_frame_equal(actual, expected)