from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_construction_with_categorical_index(self):
    ci = CategoricalIndex(list('ab') * 5, name='B')
    df = DataFrame({'A': np.random.default_rng(2).standard_normal(10), 'B': ci.values})
    idf = df.set_index('B')
    tm.assert_index_equal(idf.index, ci)
    df = DataFrame({'A': np.random.default_rng(2).standard_normal(10), 'B': ci})
    idf = df.set_index('B')
    tm.assert_index_equal(idf.index, ci)
    idf = idf.reset_index().set_index('B')
    tm.assert_index_equal(idf.index, ci)