from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_dti_union_mixed(self):
    rng = DatetimeIndex([Timestamp('2011-01-01'), pd.NaT])
    rng2 = DatetimeIndex(['2012-01-01', '2012-01-02'], tz='Asia/Tokyo')
    result = rng.union(rng2)
    expected = Index([Timestamp('2011-01-01'), pd.NaT, Timestamp('2012-01-01', tz='Asia/Tokyo'), Timestamp('2012-01-02', tz='Asia/Tokyo')], dtype=object)
    tm.assert_index_equal(result, expected)