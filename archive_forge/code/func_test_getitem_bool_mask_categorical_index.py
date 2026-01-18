import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_bool_mask_categorical_index(self):
    df3 = DataFrame({'A': np.arange(6, dtype='int64')}, index=CategoricalIndex([1, 1, 2, 1, 3, 2], dtype=CategoricalDtype([3, 2, 1], ordered=True), name='B'))
    df4 = DataFrame({'A': np.arange(6, dtype='int64')}, index=CategoricalIndex([1, 1, 2, 1, 3, 2], dtype=CategoricalDtype([3, 2, 1], ordered=False), name='B'))
    result = df3[df3.index == 'a']
    expected = df3.iloc[[]]
    tm.assert_frame_equal(result, expected)
    result = df4[df4.index == 'a']
    expected = df4.iloc[[]]
    tm.assert_frame_equal(result, expected)
    result = df3[df3.index == 1]
    expected = df3.iloc[[0, 1, 3]]
    tm.assert_frame_equal(result, expected)
    result = df4[df4.index == 1]
    expected = df4.iloc[[0, 1, 3]]
    tm.assert_frame_equal(result, expected)
    result = df3[df3.index < 2]
    expected = df3.iloc[[4]]
    tm.assert_frame_equal(result, expected)
    result = df3[df3.index > 1]
    expected = df3.iloc[[]]
    tm.assert_frame_equal(result, expected)
    msg = 'Unordered Categoricals can only compare equality or not'
    with pytest.raises(TypeError, match=msg):
        df4[df4.index < 2]
    with pytest.raises(TypeError, match=msg):
        df4[df4.index > 1]