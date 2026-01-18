import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('dtype', [SparseDtype(int, 0), int])
def test_constructor_na_dtype(self, dtype):
    with pytest.raises(ValueError, match='Cannot convert'):
        SparseArray([0, 1, np.nan], dtype=dtype)