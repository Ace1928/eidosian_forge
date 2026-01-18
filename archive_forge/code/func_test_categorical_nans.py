from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_categorical_nans(self):
    s = Series(Categorical(list('aaaaabbbcc')))
    s.iloc[1] = np.nan
    result = s.value_counts()
    expected = Series([4, 3, 2], index=CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c']), name='count')
    tm.assert_series_equal(result, expected, check_index_type=True)
    result = s.value_counts(dropna=False)
    expected = Series([4, 3, 2, 1], index=CategoricalIndex(['a', 'b', 'c', np.nan]), name='count')
    tm.assert_series_equal(result, expected, check_index_type=True)
    s = Series(Categorical(list('aaaaabbbcc'), ordered=True, categories=['b', 'a', 'c']))
    s.iloc[1] = np.nan
    result = s.value_counts()
    expected = Series([4, 3, 2], index=CategoricalIndex(['a', 'b', 'c'], categories=['b', 'a', 'c'], ordered=True), name='count')
    tm.assert_series_equal(result, expected, check_index_type=True)
    result = s.value_counts(dropna=False)
    expected = Series([4, 3, 2, 1], index=CategoricalIndex(['a', 'b', 'c', np.nan], categories=['b', 'a', 'c'], ordered=True), name='count')
    tm.assert_series_equal(result, expected, check_index_type=True)