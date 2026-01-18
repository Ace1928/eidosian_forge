import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('exact', [False, 'equiv'])
def test_index_equal_class(exact):
    idx1 = Index([0, 1, 2])
    idx2 = RangeIndex(3)
    tm.assert_index_equal(idx1, idx2, exact=exact)