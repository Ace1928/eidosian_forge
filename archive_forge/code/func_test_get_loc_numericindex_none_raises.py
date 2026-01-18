import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('dtype', ['f8', 'i8', 'u8'])
def test_get_loc_numericindex_none_raises(self, dtype):
    arr = np.arange(10 ** 7, dtype=dtype)
    idx = Index(arr)
    with pytest.raises(KeyError, match='None'):
        idx.get_loc(None)