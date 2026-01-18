import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
def test_categorical_index_repr_period_ordered(self):
    idx = period_range('2011-01-01 09:00', freq='h', periods=5)
    i = CategoricalIndex(Categorical(idx, ordered=True))
    exp = "CategoricalIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00',\n                  '2011-01-01 12:00', '2011-01-01 13:00'],\n                 categories=[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00], ordered=True, dtype='category')"
    assert repr(i) == exp
    idx = period_range('2011-01', freq='M', periods=5)
    i = CategoricalIndex(Categorical(idx, ordered=True))
    exp = "CategoricalIndex(['2011-01', '2011-02', '2011-03', '2011-04', '2011-05'], categories=[2011-01, 2011-02, 2011-03, 2011-04, 2011-05], ordered=True, dtype='category')"
    assert repr(i) == exp