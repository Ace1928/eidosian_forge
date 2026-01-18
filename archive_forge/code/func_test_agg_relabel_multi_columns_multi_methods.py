import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
def test_agg_relabel_multi_columns_multi_methods():
    df = pd.DataFrame({'A': [1, 2, 1, 2], 'B': [1, 2, 3, 4], 'C': [3, 4, 5, 6]})
    result = df.agg(foo=('A', 'sum'), bar=('B', 'mean'), cat=('A', 'min'), dat=('B', 'max'), f=('A', 'max'), g=('C', 'min'))
    expected = pd.DataFrame({'A': [6.0, np.nan, 1.0, np.nan, 2.0, np.nan], 'B': [np.nan, 2.5, np.nan, 4.0, np.nan, np.nan], 'C': [np.nan, np.nan, np.nan, np.nan, np.nan, 3.0]}, index=pd.Index(['foo', 'bar', 'cat', 'dat', 'f', 'g']))
    tm.assert_frame_equal(result, expected)