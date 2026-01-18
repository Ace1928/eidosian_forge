from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_with_datetime_key(self):
    df = DataFrame({'id': ['a', 'b'] * 3, 'b': date_range('2000-01-01', '2000-01-03', freq='9h')})
    grouper = Grouper(key='b', freq='D')
    gb = df.groupby([grouper, 'id'])
    expected = {(Timestamp('2000-01-01'), 'a'): [0, 2], (Timestamp('2000-01-01'), 'b'): [1], (Timestamp('2000-01-02'), 'a'): [4], (Timestamp('2000-01-02'), 'b'): [3, 5]}
    tm.assert_dict_equal(gb.groups, expected)
    assert len(gb.groups.keys()) == 4