import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_constructor_dtype_str(self):
    result = SparseArray([1, 2, 3], dtype='int')
    expected = SparseArray([1, 2, 3], dtype=int)
    tm.assert_sp_array_equal(result, expected)