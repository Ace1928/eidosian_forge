import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_astype_to_integer_array():
    arr = pd.array([0.0, 1.5, None], dtype='Float64')
    result = arr.astype('Int64')
    expected = pd.array([0, 1, None], dtype='Int64')
    tm.assert_extension_array_equal(result, expected)