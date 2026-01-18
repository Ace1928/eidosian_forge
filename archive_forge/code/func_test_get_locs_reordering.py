from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('keys,expected', [((slice(None), [5, 4]), [1, 0]), ((slice(None), [4, 5]), [0, 1]), (([True, False, True], [4, 6]), [0, 2]), (([True, False, True], [6, 4]), [0, 2]), ((2, [4, 5]), [0, 1]), ((2, [5, 4]), [1, 0]), (([2], [4, 5]), [0, 1]), (([2], [5, 4]), [1, 0])])
def test_get_locs_reordering(keys, expected):
    idx = MultiIndex.from_arrays([[2, 2, 1], [4, 5, 6]])
    result = idx.get_locs(keys)
    expected = np.array(expected, dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)