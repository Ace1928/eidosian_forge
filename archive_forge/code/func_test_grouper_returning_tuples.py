from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('func', [False, True])
def test_grouper_returning_tuples(self, func):
    df = DataFrame({'X': ['A', 'B', 'A', 'B'], 'Y': [1, 4, 3, 2]})
    mapping = dict(zip(range(4), [('C', 5), ('D', 6)] * 2))
    if func:
        gb = df.groupby(by=lambda idx: mapping[idx], sort=False)
    else:
        gb = df.groupby(by=mapping, sort=False)
    name, expected = next(iter(gb))
    assert name == ('C', 5)
    result = gb.get_group(name)
    tm.assert_frame_equal(result, expected)