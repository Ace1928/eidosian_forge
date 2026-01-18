import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ea_categories():
    df = DataFrame({'a': [1, 0, 0, 1], 'b': [0, 1, 0, 0], 'c': [0, 0, 1, 0]})
    df.columns = df.columns.astype('string[python]')
    result = from_dummies(df)
    expected = DataFrame({'': Series(list('abca'), dtype='string[python]')})
    tm.assert_frame_equal(result, expected)