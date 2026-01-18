from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
def test_nlargest_duplicates_on_starter_columns(self):
    df = pd.DataFrame({'a': [2, 2, 2, 1, 1, 1], 'b': [1, 2, 3, 3, 2, 1]})
    result = df.nlargest(4, columns=['a', 'b'])
    expected = pd.DataFrame({'a': [2, 2, 2, 1], 'b': [3, 2, 1, 3]}, index=[2, 1, 0, 3])
    tm.assert_frame_equal(result, expected)
    result = df.nsmallest(4, columns=['a', 'b'])
    expected = pd.DataFrame({'a': [1, 1, 1, 2], 'b': [1, 2, 3, 1]}, index=[5, 4, 3, 0])
    tm.assert_frame_equal(result, expected)