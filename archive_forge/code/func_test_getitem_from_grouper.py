from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('func', [lambda x: x.sum(), lambda x: x.agg(lambda y: y.sum())])
def test_getitem_from_grouper(self, func):
    df = DataFrame({'a': [1, 1, 2], 'b': 3, 'c': 4, 'd': 5})
    gb = df.groupby(['a', 'b'])[['a', 'c']]
    idx = MultiIndex.from_tuples([(1, 3), (2, 3)], names=['a', 'b'])
    expected = DataFrame({'a': [2, 2], 'c': [8, 4]}, index=idx)
    result = func(gb)
    tm.assert_frame_equal(result, expected)