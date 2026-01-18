import inspect
import operator
import numpy as np
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort
def test_combine_add(self, data_repeated):
    orig_data1, orig_data2 = data_repeated(2)
    s1 = pd.Series(orig_data1)
    s2 = pd.Series(orig_data2)
    try:
        with np.errstate(over='ignore'):
            expected = pd.Series(orig_data1._from_sequence([a + b for a, b in zip(list(orig_data1), list(orig_data2))]))
    except TypeError:
        with pytest.raises(TypeError):
            s1.combine(s2, lambda x1, x2: x1 + x2)
        return
    result = s1.combine(s2, lambda x1, x2: x1 + x2)
    tm.assert_series_equal(result, expected)
    val = s1.iloc[0]
    result = s1.combine(val, lambda x1, x2: x1 + x2)
    expected = pd.Series(orig_data1._from_sequence([a + val for a in list(orig_data1)]))
    tm.assert_series_equal(result, expected)