from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
@pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
def test_interval_can_hold_element(self, dtype, element):
    arr = np.array([1, 3, 4, 9], dtype=dtype)
    ii = IntervalIndex.from_breaks(arr)
    blk = new_block(ii._data, BlockPlacement([1]), ndim=2)
    elem = element(ii)
    self.check_series_setitem(elem, ii, True)
    assert blk._can_hold_element(elem)
    ii2 = IntervalIndex.from_breaks(arr[:-1], closed='neither')
    elem = element(ii2)
    with tm.assert_produces_warning(FutureWarning):
        self.check_series_setitem(elem, ii, False)
    assert not blk._can_hold_element(elem)
    ii3 = IntervalIndex.from_breaks([Timestamp(1), Timestamp(3), Timestamp(4)])
    elem = element(ii3)
    with tm.assert_produces_warning(FutureWarning):
        self.check_series_setitem(elem, ii, False)
    assert not blk._can_hold_element(elem)
    ii4 = IntervalIndex.from_breaks([Timedelta(1), Timedelta(3), Timedelta(4)])
    elem = element(ii4)
    with tm.assert_produces_warning(FutureWarning):
        self.check_series_setitem(elem, ii, False)
    assert not blk._can_hold_element(elem)