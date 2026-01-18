import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
@pytest.mark.parametrize('values', [['1', '2', None], ['1.5', '2', None]])
def test_construct_from_float_strings(values):
    expected = pd.array([float(values[0]), 2, None], dtype='Float64')
    res = pd.array(values, dtype='Float64')
    tm.assert_extension_array_equal(res, expected)
    res = FloatingArray._from_sequence(values)
    tm.assert_extension_array_equal(res, expected)