import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index,val', [(Index([0, 1, 2]), '2'), (Index([0, 1, '2']), 2), (Index([0, 1, 2, np.inf]), 4), (Index([0, 1, 2, np.nan]), 4), (Index([0, 1, 2, np.inf]), np.nan), (Index([0, 1, 2, np.nan]), np.inf), (Index([0, 1, 2], dtype=np.int64), np.inf), (Index([0, 1, 2], dtype=np.int64), np.nan), (Index([0, 1, 2], dtype=np.uint64), np.inf), (Index([0, 1, 2], dtype=np.uint64), np.nan)])
def test_index_not_contains(self, index, val):
    assert val not in index