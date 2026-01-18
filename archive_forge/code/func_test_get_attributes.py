import string
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('attr', ['npoints', 'density', 'fill_value', 'sp_values'])
def test_get_attributes(self, attr):
    arr = SparseArray([0, 1])
    ser = pd.Series(arr)
    result = getattr(ser.sparse, attr)
    expected = getattr(arr, attr)
    assert result == expected