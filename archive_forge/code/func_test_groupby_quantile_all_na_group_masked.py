import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('interpolation, val1, val2', [('lower', 2, 2), ('higher', 2, 3), ('nearest', 2, 2)])
def test_groupby_quantile_all_na_group_masked(interpolation, val1, val2, any_numeric_ea_dtype):
    df = DataFrame({'a': [1, 1, 1, 2], 'b': [1, 2, 3, pd.NA]}, dtype=any_numeric_ea_dtype)
    result = df.groupby('a').quantile(q=[0.5, 0.7], interpolation=interpolation)
    expected = DataFrame({'b': [val1, val2, pd.NA, pd.NA]}, dtype=any_numeric_ea_dtype, index=pd.MultiIndex.from_arrays([pd.Series([1, 1, 2, 2], dtype=any_numeric_ea_dtype), [0.5, 0.7, 0.5, 0.7]], names=['a', None]))
    tm.assert_frame_equal(result, expected)