import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('index', [pd.RangeIndex(0, 4), list('abcd'), pd.MultiIndex.from_product([(1, 2), ('R', 'B')], names=['num', 'col'])])
def test_groupby_dataframe_slice_then_transform(dropna, index):
    expected_data = {'B': [2, 2, 1, np.nan if dropna else 1]}
    df = pd.DataFrame({'A': [0, 0, 1, None], 'B': [1, 2, 3, None]}, index=index)
    gb = df.groupby('A', dropna=dropna)
    result = gb.transform(len)
    expected = pd.DataFrame(expected_data, index=index)
    tm.assert_frame_equal(result, expected)
    result = gb[['B']].transform(len)
    expected = pd.DataFrame(expected_data, index=index)
    tm.assert_frame_equal(result, expected)
    result = gb['B'].transform(len)
    expected = pd.Series(expected_data['B'], index=index, name='B')
    tm.assert_series_equal(result, expected)