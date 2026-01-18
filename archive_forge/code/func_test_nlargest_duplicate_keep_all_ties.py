from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
def test_nlargest_duplicate_keep_all_ties(self):
    df = pd.DataFrame({'a': [5, 4, 4, 2, 3, 3, 3, 3], 'b': [10, 9, 8, 7, 5, 50, 10, 20]})
    result = df.nlargest(4, 'a', keep='all')
    expected = pd.DataFrame({'a': {0: 5, 1: 4, 2: 4, 4: 3, 5: 3, 6: 3, 7: 3}, 'b': {0: 10, 1: 9, 2: 8, 4: 5, 5: 50, 6: 10, 7: 20}})
    tm.assert_frame_equal(result, expected)
    result = df.nsmallest(2, 'a', keep='all')
    expected = pd.DataFrame({'a': {3: 2, 4: 3, 5: 3, 6: 3, 7: 3}, 'b': {3: 7, 4: 5, 5: 50, 6: 10, 7: 20}})
    tm.assert_frame_equal(result, expected)