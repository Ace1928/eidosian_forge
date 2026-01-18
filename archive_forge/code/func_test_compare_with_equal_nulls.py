import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
def test_compare_with_equal_nulls():
    df = pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': [1.0, 2.0, np.nan], 'col3': [1.0, 2.0, 3.0]}, columns=['col1', 'col2', 'col3'])
    df2 = df.copy()
    df2.loc[0, 'col1'] = 'c'
    result = df.compare(df2)
    indices = pd.Index([0])
    columns = pd.MultiIndex.from_product([['col1'], ['self', 'other']])
    expected = pd.DataFrame([['a', 'c']], index=indices, columns=columns)
    tm.assert_frame_equal(result, expected)