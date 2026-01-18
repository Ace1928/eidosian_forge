from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
@pytest.mark.parametrize('kwargs', [{}, {'other': None}])
def test_df_where_with_category(self, kwargs):
    data = np.arange(2 * 3, dtype=np.int64).reshape(2, 3)
    df = DataFrame(data, columns=list('ABC'))
    mask = np.array([[True, False, False], [False, False, True]])
    df.A = df.A.astype('category')
    df.B = df.B.astype('category')
    df.C = df.C.astype('category')
    result = df.where(mask, **kwargs)
    A = pd.Categorical([0, np.nan], categories=[0, 3])
    B = pd.Categorical([np.nan, np.nan], categories=[1, 4])
    C = pd.Categorical([np.nan, 5], categories=[2, 5])
    expected = DataFrame({'A': A, 'B': B, 'C': C})
    tm.assert_frame_equal(result, expected)
    result = df.A.where(mask[:, 0], **kwargs)
    expected = Series(A, name='A')
    tm.assert_series_equal(result, expected)