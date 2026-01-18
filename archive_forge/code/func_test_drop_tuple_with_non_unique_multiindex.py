import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('indexer', [('a', 'a'), [('a', 'a')]])
def test_drop_tuple_with_non_unique_multiindex(self, indexer):
    idx = MultiIndex.from_product([['a', 'b'], ['a', 'a']])
    df = DataFrame({'x': range(len(idx))}, index=idx)
    result = df.drop(index=[('a', 'a')])
    expected = DataFrame({'x': [2, 3]}, index=MultiIndex.from_tuples([('b', 'a'), ('b', 'a')]))
    tm.assert_frame_equal(result, expected)