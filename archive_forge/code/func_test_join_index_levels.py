import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_join_index_levels():
    midx = midx = MultiIndex.from_tuples([('a', '2019-02-01'), ('a', '2019-02-01')])
    midx2 = MultiIndex.from_tuples([('a', '2019-01-31')])
    result = midx.join(midx2, how='outer')
    expected = MultiIndex.from_tuples([('a', '2019-01-31'), ('a', '2019-02-01'), ('a', '2019-02-01')])
    tm.assert_index_equal(result.levels[1], expected.levels[1])
    tm.assert_index_equal(result, expected)