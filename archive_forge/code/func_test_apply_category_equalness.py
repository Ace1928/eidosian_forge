from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('val', ['asd', 12, None, np.nan])
def test_apply_category_equalness(val):
    df_values = ['asd', None, 12, 'asd', 'cde', np.nan]
    df = DataFrame({'a': df_values}, dtype='category')
    result = df.a.apply(lambda x: x == val)
    expected = Series([np.nan if pd.isnull(x) else x == val for x in df_values], name='a')
    tm.assert_series_equal(result, expected)