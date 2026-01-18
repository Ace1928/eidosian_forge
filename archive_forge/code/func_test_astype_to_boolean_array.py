import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_astype_to_boolean_array():
    arr = pd.array([0.0, 1.0, None], dtype='Float64')
    result = arr.astype('boolean')
    expected = pd.array([False, True, None], dtype='boolean')
    tm.assert_extension_array_equal(result, expected)
    result = arr.astype(pd.BooleanDtype())
    tm.assert_extension_array_equal(result, expected)