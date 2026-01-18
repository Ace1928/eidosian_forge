import numpy as np
import pytest
from pandas import (
from pandas.core.arrays.sparse import SparseArray
def test_bool_sum_min_count(self):
    spar_bool = SparseArray([False, True] * 5, dtype=np.bool_, fill_value=True)
    res = spar_bool.sum(min_count=1)
    assert res == 5
    res = spar_bool.sum(min_count=11)
    assert isna(res)