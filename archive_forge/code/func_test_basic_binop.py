import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
def test_basic_binop():
    x = NumpyExtensionArray(np.array([1, 2, 3]))
    result = x + x
    expected = NumpyExtensionArray(np.array([2, 4, 6]))
    tm.assert_extension_array_equal(result, expected)