import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.indexes.accessors import Properties
def test_cat_accessor_api(self):
    assert Series.cat is CategoricalAccessor
    ser = Series(list('aabbcde')).astype('category')
    assert isinstance(ser.cat, CategoricalAccessor)
    invalid = Series([1])
    with pytest.raises(AttributeError, match='only use .cat accessor'):
        invalid.cat
    assert not hasattr(invalid, 'cat')