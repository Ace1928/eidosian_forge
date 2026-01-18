from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a, b, expected_names', [(Series([1, 2, 3], name='foo'), Series(['a', 'b'], name='bar'), ['foo', 'bar']), (Series([1, 2, 3], name='foo'), ['a', 'b'], ['foo', None]), ([1, 2, 3], ['a', 'b'], None)])
def test_from_product_infer_names(a, b, expected_names):
    result = MultiIndex.from_product([a, b])
    expected = MultiIndex(levels=[[1, 2, 3], ['a', 'b']], codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]], names=expected_names)
    tm.assert_index_equal(result, expected)