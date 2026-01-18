import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.arrays.boolean import coerce_to_array
def test_coerce_to_array():
    values = np.array([True, False, True, False], dtype='bool')
    mask = np.array([False, False, False, True], dtype='bool')
    result = BooleanArray(*coerce_to_array(values, mask=mask))
    expected = BooleanArray(values, mask)
    tm.assert_extension_array_equal(result, expected)
    assert result._data is values
    assert result._mask is mask
    result = BooleanArray(*coerce_to_array(values, mask=mask, copy=True))
    expected = BooleanArray(values, mask)
    tm.assert_extension_array_equal(result, expected)
    assert result._data is not values
    assert result._mask is not mask
    values = [True, False, None, False]
    mask = np.array([False, False, False, True], dtype='bool')
    result = BooleanArray(*coerce_to_array(values, mask=mask))
    expected = BooleanArray(np.array([True, False, True, True]), np.array([False, False, True, True]))
    tm.assert_extension_array_equal(result, expected)
    result = BooleanArray(*coerce_to_array(np.array(values, dtype=object), mask=mask))
    tm.assert_extension_array_equal(result, expected)
    result = BooleanArray(*coerce_to_array(values, mask=mask.tolist()))
    tm.assert_extension_array_equal(result, expected)
    values = np.array([True, False, True, False], dtype='bool')
    mask = np.array([False, False, False, True], dtype='bool')
    coerce_to_array(values.reshape(1, -1))
    with pytest.raises(ValueError, match='values.shape and mask.shape must match'):
        coerce_to_array(values.reshape(1, -1), mask=mask)
    with pytest.raises(ValueError, match='values.shape and mask.shape must match'):
        coerce_to_array(values, mask=mask.reshape(1, -1))