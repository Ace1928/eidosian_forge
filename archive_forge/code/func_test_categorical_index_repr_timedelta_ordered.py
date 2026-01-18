import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
def test_categorical_index_repr_timedelta_ordered(self):
    idx = timedelta_range('1 days', periods=5)
    i = CategoricalIndex(Categorical(idx, ordered=True))
    exp = "CategoricalIndex(['1 days', '2 days', '3 days', '4 days', '5 days'], categories=[1 days, 2 days, 3 days, 4 days, 5 days], ordered=True, dtype='category')"
    assert repr(i) == exp
    idx = timedelta_range('1 hours', periods=10)
    i = CategoricalIndex(Categorical(idx, ordered=True))
    exp = "CategoricalIndex(['0 days 01:00:00', '1 days 01:00:00', '2 days 01:00:00',\n                  '3 days 01:00:00', '4 days 01:00:00', '5 days 01:00:00',\n                  '6 days 01:00:00', '7 days 01:00:00', '8 days 01:00:00',\n                  '9 days 01:00:00'],\n                 categories=[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, ..., 6 days 01:00:00, 7 days 01:00:00, 8 days 01:00:00, 9 days 01:00:00], ordered=True, dtype='category')"
    assert repr(i) == exp