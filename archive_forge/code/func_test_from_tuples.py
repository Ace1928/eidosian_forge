from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_tuples():
    msg = 'Cannot infer number of levels from empty list'
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_tuples([])
    expected = MultiIndex(levels=[[1, 3], [2, 4]], codes=[[0, 1], [0, 1]], names=['a', 'b'])
    result = MultiIndex.from_tuples(((1, 2), (3, 4)), names=['a', 'b'])
    tm.assert_index_equal(result, expected)