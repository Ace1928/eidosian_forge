import pytest
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm
def test_index_name_retained():
    result = pd.DataFrame({'x': [1, 2, 6], 'y': [2, 2, 8], 'z': [-5, 0, 5]})
    result = result.set_index('z')
    result.loc[10] = [9, 10]
    df_expected = pd.DataFrame({'x': [1, 2, 6, 9], 'y': [2, 2, 8, 10], 'z': [-5, 0, 5, 10]})
    df_expected = df_expected.set_index('z')
    tm.assert_frame_equal(result, df_expected)