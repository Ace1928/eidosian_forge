import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('dtype', [np.float64, np.int64, np.uint64])
def test_contains_none(self, dtype):
    index = Index([0, 1, 2, 3, 4], dtype=dtype)
    assert None not in index