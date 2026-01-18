import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_data_frame_value_counts_default():
    df = pd.DataFrame({'num_legs': [2, 4, 4, 6], 'num_wings': [2, 0, 0, 0]}, index=['falcon', 'dog', 'cat', 'ant'])
    result = df.value_counts()
    expected = pd.Series(data=[2, 1, 1], index=pd.MultiIndex.from_arrays([(4, 2, 6), (0, 2, 0)], names=['num_legs', 'num_wings']), name='count')
    tm.assert_series_equal(result, expected)