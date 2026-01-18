from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_series_repr_timedelta_ordered(self):
    idx = timedelta_range('1 days', periods=5)
    s = Series(Categorical(idx, ordered=True))
    exp = '0   1 days\n1   2 days\n2   3 days\n3   4 days\n4   5 days\ndtype: category\nCategories (5, timedelta64[ns]): [1 days < 2 days < 3 days < 4 days < 5 days]'
    assert repr(s) == exp
    idx = timedelta_range('1 hours', periods=10)
    s = Series(Categorical(idx, ordered=True))
    exp = '0   0 days 01:00:00\n1   1 days 01:00:00\n2   2 days 01:00:00\n3   3 days 01:00:00\n4   4 days 01:00:00\n5   5 days 01:00:00\n6   6 days 01:00:00\n7   7 days 01:00:00\n8   8 days 01:00:00\n9   9 days 01:00:00\ndtype: category\nCategories (10, timedelta64[ns]): [0 days 01:00:00 < 1 days 01:00:00 < 2 days 01:00:00 <\n                                   3 days 01:00:00 ... 6 days 01:00:00 < 7 days 01:00:00 <\n                                   8 days 01:00:00 < 9 days 01:00:00]'
    assert repr(s) == exp