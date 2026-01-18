import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['first', 'last'])
@pytest.mark.parametrize('df, expected', [(DataFrame({'id': 'a', 'value': [None, 'foo', np.nan]}), DataFrame({'value': ['foo']}, index=Index(['a'], name='id'))), (DataFrame({'id': 'a', 'value': [np.nan]}, dtype=object), DataFrame({'value': [None]}, index=Index(['a'], name='id')))])
def test_first_last_with_None_expanded(method, df, expected):
    result = getattr(df.groupby('id'), method)()
    tm.assert_frame_equal(result, expected)