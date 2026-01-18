from collections import deque
import re
import string
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
@pytest.mark.parametrize('ufunc', [np.divmod])
@pytest.mark.parametrize('shuffle', [True, False])
@pytest.mark.filterwarnings('ignore:divide by zero:RuntimeWarning')
def test_multiple_output_binary_ufuncs(ufunc, sparse, shuffle, arrays_for_binary_ufunc):
    a1, a2 = arrays_for_binary_ufunc
    a1[a1 == 0] = 1
    a2[a2 == 0] = 1
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype('int64', 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype('int64', 0))
    s1 = pd.Series(a1)
    s2 = pd.Series(a2)
    if shuffle:
        s2 = s2.sample(frac=1)
    expected = ufunc(a1, a2)
    assert isinstance(expected, tuple)
    result = ufunc(s1, s2)
    assert isinstance(result, tuple)
    tm.assert_series_equal(result[0], pd.Series(expected[0]))
    tm.assert_series_equal(result[1], pd.Series(expected[1]))