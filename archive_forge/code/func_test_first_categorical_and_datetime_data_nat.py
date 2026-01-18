import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_first_categorical_and_datetime_data_nat():
    df = DataFrame({'group': ['first', 'first', 'second', 'third', 'third'], 'time': 5 * [np.datetime64('NaT')], 'categories': Series(['a', 'b', 'c', 'a', 'b'], dtype='category')})
    result = df.groupby('group').first()
    expected = DataFrame({'time': 3 * [np.datetime64('NaT')], 'categories': Series(['a', 'c', 'a']).astype(pd.CategoricalDtype(['a', 'b', 'c']))})
    expected.index = Index(['first', 'second', 'third'], name='group')
    tm.assert_frame_equal(result, expected)