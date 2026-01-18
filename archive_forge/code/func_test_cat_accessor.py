import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.indexes.accessors import Properties
def test_cat_accessor(self):
    ser = Series(Categorical(['a', 'b', np.nan, 'a']))
    tm.assert_index_equal(ser.cat.categories, Index(['a', 'b']))
    assert not ser.cat.ordered, False
    exp = Categorical(['a', 'b', np.nan, 'a'], categories=['b', 'a'])
    res = ser.cat.set_categories(['b', 'a'])
    tm.assert_categorical_equal(res.values, exp)
    ser[:] = 'a'
    ser = ser.cat.remove_unused_categories()
    tm.assert_index_equal(ser.cat.categories, Index(['a']))