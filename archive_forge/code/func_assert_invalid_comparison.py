import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def assert_invalid_comparison(left, right, box):
    """
    Assert that comparison operations with mismatched types behave correctly.

    Parameters
    ----------
    left : np.ndarray, ExtensionArray, Index, or Series
    right : object
    box : {pd.DataFrame, pd.Series, pd.Index, pd.array, tm.to_array}
    """
    xbox = box if box not in [Index, array] else np.array

    def xbox2(x):
        if isinstance(x, NumpyExtensionArray):
            return x._ndarray
        if isinstance(x, BooleanArray):
            return x.astype(bool)
        return x
    rev_box = xbox
    if isinstance(right, Index) and isinstance(left, Series):
        rev_box = np.array
    result = xbox2(left == right)
    expected = xbox(np.zeros(result.shape, dtype=np.bool_))
    tm.assert_equal(result, expected)
    result = xbox2(right == left)
    tm.assert_equal(result, rev_box(expected))
    result = xbox2(left != right)
    tm.assert_equal(result, ~expected)
    result = xbox2(right != left)
    tm.assert_equal(result, rev_box(~expected))
    msg = '|'.join(['Invalid comparison between', 'Cannot compare type', 'not supported between', 'invalid type promotion', "The DTypes <class 'numpy.dtype\\[datetime64\\]'> and <class 'numpy.dtype\\[int64\\]'> do not have a common DType. For example they cannot be stored in a single array unless the dtype is `object`."])
    with pytest.raises(TypeError, match=msg):
        left < right
    with pytest.raises(TypeError, match=msg):
        left <= right
    with pytest.raises(TypeError, match=msg):
        left > right
    with pytest.raises(TypeError, match=msg):
        left >= right
    with pytest.raises(TypeError, match=msg):
        right < left
    with pytest.raises(TypeError, match=msg):
        right <= left
    with pytest.raises(TypeError, match=msg):
        right > left
    with pytest.raises(TypeError, match=msg):
        right >= left