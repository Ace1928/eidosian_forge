import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
def test_categorical_index_repr_datetime_ordered(self):
    idx = date_range('2011-01-01 09:00', freq='h', periods=5)
    i = CategoricalIndex(Categorical(idx, ordered=True))
    exp = "CategoricalIndex(['2011-01-01 09:00:00', '2011-01-01 10:00:00',\n                  '2011-01-01 11:00:00', '2011-01-01 12:00:00',\n                  '2011-01-01 13:00:00'],\n                 categories=[2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00, 2011-01-01 12:00:00, 2011-01-01 13:00:00], ordered=True, dtype='category')"
    assert repr(i) == exp
    idx = date_range('2011-01-01 09:00', freq='h', periods=5, tz='US/Eastern')
    i = CategoricalIndex(Categorical(idx, ordered=True))
    exp = "CategoricalIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00',\n                  '2011-01-01 11:00:00-05:00', '2011-01-01 12:00:00-05:00',\n                  '2011-01-01 13:00:00-05:00'],\n                 categories=[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00], ordered=True, dtype='category')"
    assert repr(i) == exp
    i = CategoricalIndex(Categorical(idx.append(idx), ordered=True))
    exp = "CategoricalIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00',\n                  '2011-01-01 11:00:00-05:00', '2011-01-01 12:00:00-05:00',\n                  '2011-01-01 13:00:00-05:00', '2011-01-01 09:00:00-05:00',\n                  '2011-01-01 10:00:00-05:00', '2011-01-01 11:00:00-05:00',\n                  '2011-01-01 12:00:00-05:00', '2011-01-01 13:00:00-05:00'],\n                 categories=[2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00, 2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00, 2011-01-01 13:00:00-05:00], ordered=True, dtype='category')"
    assert repr(i) == exp