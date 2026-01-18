from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_scalar_on_date_time_index_aware_series():
    series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10, tz='UTC'), name='ts')
    result = Series(series.index).map(lambda x: 1)
    tm.assert_series_equal(result, Series(np.ones(len(series)), dtype='int64'))