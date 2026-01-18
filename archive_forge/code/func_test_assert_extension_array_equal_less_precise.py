import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('decimals', range(10))
def test_assert_extension_array_equal_less_precise(decimals):
    rtol = 0.5 * 10 ** (-decimals)
    arr1 = SparseArray([0.5, 0.123456])
    arr2 = SparseArray([0.5, 0.123457])
    if decimals >= 5:
        msg = 'ExtensionArray are different\n\nExtensionArray values are different \\(50\\.0 %\\)\n\\[left\\]:  \\[0\\.5, 0\\.123456\\]\n\\[right\\]: \\[0\\.5, 0\\.123457\\]'
        with pytest.raises(AssertionError, match=msg):
            tm.assert_extension_array_equal(arr1, arr2, rtol=rtol)
    else:
        tm.assert_extension_array_equal(arr1, arr2, rtol=rtol)