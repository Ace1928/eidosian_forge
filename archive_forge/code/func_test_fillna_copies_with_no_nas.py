import numpy as np
import pytest
from pandas import CategoricalIndex
import pandas._testing as tm
def test_fillna_copies_with_no_nas(self):
    ci = CategoricalIndex([0, 1, 1])
    result = ci.fillna(0)
    assert result is not ci
    assert tm.shares_memory(result, ci)
    cat = ci._data
    result = cat.fillna(0)
    assert result._ndarray is not cat._ndarray
    assert result._ndarray.base is None
    assert not tm.shares_memory(result, cat)