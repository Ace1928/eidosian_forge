import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ea_categories_with_sep():
    df = DataFrame({'col1_a': [1, 0, 1], 'col1_b': [0, 1, 0], 'col2_a': [0, 1, 0], 'col2_b': [1, 0, 0], 'col2_c': [0, 0, 1]})
    df.columns = df.columns.astype('string[python]')
    result = from_dummies(df, sep='_')
    expected = DataFrame({'col1': Series(list('aba'), dtype='string[python]'), 'col2': Series(list('bac'), dtype='string[python]')})
    expected.columns = expected.columns.astype('string[python]')
    tm.assert_frame_equal(result, expected)