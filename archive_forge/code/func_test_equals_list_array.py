from contextlib import nullcontext
import copy
import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import is_float
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val', [1, 1.1, 1 + 1j, True, 'abc', [1, 2], (1, 2), {1, 2}, {'a': 1}, None])
def test_equals_list_array(val):
    arr = np.array([1, 2])
    s1 = Series([arr, arr])
    s2 = s1.copy()
    assert s1.equals(s2)
    s1[1] = val
    cm = tm.assert_produces_warning(FutureWarning, check_stacklevel=False) if isinstance(val, str) and (not np_version_gte1p25) else nullcontext()
    with cm:
        assert not s1.equals(s2)