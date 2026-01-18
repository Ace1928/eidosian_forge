import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('exact', [False, True])
def test_index_equal_range_categories(check_categorical, exact):
    msg = "Index are different\n\nIndex classes are different\n\\[left\\]:  RangeIndex\\(start=0, stop=10, step=1\\)\n\\[right\\]: Index\\(\\[0, 1, 2, 3, 4, 5, 6, 7, 8, 9\\], dtype='int64'\\)"
    rcat = CategoricalIndex(RangeIndex(10))
    icat = CategoricalIndex(list(range(10)))
    if check_categorical and exact:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(rcat, icat, check_categorical=True, exact=True)
    else:
        tm.assert_index_equal(rcat, icat, check_categorical=check_categorical, exact=exact)