import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('func', ['ffill', 'bfill'])
def test_groupby_column_index_name_lost_fill_funcs(func):
    df = DataFrame([[1, 1.0, -1.0], [1, np.nan, np.nan], [1, 2.0, -2.0]], columns=Index(['type', 'a', 'b'], name='idx'))
    df_grouped = df.groupby(['type'])[['a', 'b']]
    result = getattr(df_grouped, func)().columns
    expected = Index(['a', 'b'], name='idx')
    tm.assert_index_equal(result, expected)