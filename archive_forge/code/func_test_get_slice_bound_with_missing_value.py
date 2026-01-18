from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index_arr,expected,target,algo', [([[np.nan, 'a', 'b'], ['c', 'd', 'e']], 0, np.nan, 'left'), ([[np.nan, 'a', 'b'], ['c', 'd', 'e']], 1, (np.nan, 'c'), 'right'), ([['a', 'b', 'c'], ['d', np.nan, 'd']], 1, ('b', np.nan), 'left')])
def test_get_slice_bound_with_missing_value(index_arr, expected, target, algo):
    idx = MultiIndex.from_arrays(index_arr)
    result = idx.get_slice_bound(target, side=algo)
    assert result == expected