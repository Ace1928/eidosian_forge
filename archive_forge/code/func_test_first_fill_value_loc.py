import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('arr,fill_value,loc', [([None, 1, 2], None, 0), ([0, None, 2], None, 1), ([0, 1, None], None, 2), ([0, 1, 1, None, None], None, 3), ([1, 1, 1, 2], None, -1), ([], None, -1), ([None, 1, 0, 0, None, 2], None, 0), ([None, 1, 0, 0, None, 2], 1, 1), ([None, 1, 0, 0, None, 2], 2, 5), ([None, 1, 0, 0, None, 2], 3, -1), ([None, 0, 0, 1, 2, 1], 0, 1), ([None, 0, 0, 1, 2, 1], 1, 3)])
def test_first_fill_value_loc(arr, fill_value, loc):
    result = SparseArray(arr, fill_value=fill_value)._first_fill_value_loc()
    assert result == loc