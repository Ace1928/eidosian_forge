import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('side', ['left', 'right'])
def test_assert_extension_array_equal_non_extension_array(side):
    numpy_array = np.arange(5)
    extension_array = SparseArray(numpy_array)
    msg = f'{side} is not an ExtensionArray'
    args = (numpy_array, extension_array) if side == 'left' else (extension_array, numpy_array)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_extension_array_equal(*args)