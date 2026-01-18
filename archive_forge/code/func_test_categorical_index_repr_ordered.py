import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
def test_categorical_index_repr_ordered(self):
    i = CategoricalIndex(Categorical([1, 2, 3], ordered=True))
    exp = "CategoricalIndex([1, 2, 3], categories=[1, 2, 3], ordered=True, dtype='category')"
    assert repr(i) == exp
    i = CategoricalIndex(Categorical(np.arange(10, dtype=np.int64), ordered=True))
    exp = "CategoricalIndex([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], categories=[0, 1, 2, 3, ..., 6, 7, 8, 9], ordered=True, dtype='category')"
    assert repr(i) == exp