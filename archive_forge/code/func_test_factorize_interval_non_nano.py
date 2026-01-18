from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_factorize_interval_non_nano(self, unit):
    left = DatetimeIndex(['2016-01-01', np.nan, '2015-10-11']).as_unit(unit)
    right = DatetimeIndex(['2016-01-02', np.nan, '2015-10-15']).as_unit(unit)
    idx = IntervalIndex.from_arrays(left, right)
    codes, cats = idx.factorize()
    assert cats.dtype == f'interval[datetime64[{unit}], right]'
    ts = Timestamp(0).as_unit(unit)
    idx2 = IntervalIndex.from_arrays(left - ts, right - ts)
    codes2, cats2 = idx2.factorize()
    assert cats2.dtype == f'interval[timedelta64[{unit}], right]'
    idx3 = IntervalIndex.from_arrays(left.tz_localize('US/Pacific'), right.tz_localize('US/Pacific'))
    codes3, cats3 = idx3.factorize()
    assert cats3.dtype == f'interval[datetime64[{unit}, US/Pacific], right]'