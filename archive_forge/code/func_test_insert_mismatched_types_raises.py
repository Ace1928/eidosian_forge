from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('item', [0, np.int64(0), np.float64(0), np.array(0), np.timedelta64(456)])
def test_insert_mismatched_types_raises(self, tz_aware_fixture, item):
    tz = tz_aware_fixture
    dti = date_range('2019-11-04', periods=9, freq='-1D', name=9, tz=tz)
    result = dti.insert(1, item)
    if isinstance(item, np.ndarray):
        assert item.item() == 0
        expected = Index([dti[0], 0] + list(dti[1:]), dtype=object, name=9)
    else:
        expected = Index([dti[0], item] + list(dti[1:]), dtype=object, name=9)
    tm.assert_index_equal(result, expected)