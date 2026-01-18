from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [bool, object])
def test_get_loc_cast_bool(self, dtype):
    levels = [Index([False, True], dtype=dtype), np.arange(2, dtype='int64')]
    idx = MultiIndex.from_product(levels)
    if dtype is bool:
        with pytest.raises(KeyError, match='^\\(0, 1\\)$'):
            assert idx.get_loc((0, 1)) == 1
        with pytest.raises(KeyError, match='^\\(1, 0\\)$'):
            assert idx.get_loc((1, 0)) == 2
    else:
        assert idx.get_loc((0, 1)) == 1
        assert idx.get_loc((1, 0)) == 2
    with pytest.raises(KeyError, match='^\\(False, True\\)$'):
        idx.get_loc((False, True))
    with pytest.raises(KeyError, match='^\\(True, False\\)$'):
        idx.get_loc((True, False))