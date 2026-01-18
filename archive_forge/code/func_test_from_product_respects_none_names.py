from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_product_respects_none_names():
    a = Series([1, 2, 3], name='foo')
    b = Series(['a', 'b'], name='bar')
    result = MultiIndex.from_product([a, b], names=None)
    expected = MultiIndex(levels=[[1, 2, 3], ['a', 'b']], codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]], names=None)
    tm.assert_index_equal(result, expected)