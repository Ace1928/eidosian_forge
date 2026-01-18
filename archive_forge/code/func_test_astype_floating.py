import numpy as np
import pytest
from pandas.core.dtypes.generic import ABCIndex
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import (
def test_astype_floating():
    arr = pd.array([1, 2, None], dtype='Int64')
    result = arr.astype('Float64')
    expected = pd.array([1.0, 2.0, None], dtype='Float64')
    tm.assert_extension_array_equal(result, expected)