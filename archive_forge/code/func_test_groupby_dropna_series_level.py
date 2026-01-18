import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('dropna, idx, expected', [(True, ['a', 'a', 'b', np.nan], pd.Series([3, 3], index=['a', 'b'])), (False, ['a', 'a', 'b', np.nan], pd.Series([3, 3, 3], index=['a', 'b', np.nan]))])
def test_groupby_dropna_series_level(dropna, idx, expected):
    ser = pd.Series([1, 2, 3, 3], index=idx)
    result = ser.groupby(level=0, dropna=dropna).sum()
    tm.assert_series_equal(result, expected)