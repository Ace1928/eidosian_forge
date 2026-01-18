import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_integer
from pandas.core.arrays import IntegerArray
from pandas.core.arrays.integer import (
def test_integer_array_from_boolean():
    expected = pd.array(np.array([True, False]), dtype='Int64')
    result = pd.array(np.array([True, False], dtype=object), dtype='Int64')
    tm.assert_extension_array_equal(result, expected)