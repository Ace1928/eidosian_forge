import numpy as np
import pytest
from pandas import (
from pandas.tests.io.pytables.common import (
def test_categorical_nan_only_columns(tmp_path, setup_path):
    df = DataFrame({'a': ['a', 'b', 'c', np.nan], 'b': [np.nan, np.nan, np.nan, np.nan], 'c': [1, 2, 3, 4], 'd': Series([None] * 4, dtype=object)})
    df['a'] = df.a.astype('category')
    df['b'] = df.b.astype('category')
    df['d'] = df.b.astype('category')
    expected = df
    path = tmp_path / setup_path
    df.to_hdf(path, key='df', format='table', data_columns=True)
    result = read_hdf(path, 'df')
    tm.assert_frame_equal(result, expected)