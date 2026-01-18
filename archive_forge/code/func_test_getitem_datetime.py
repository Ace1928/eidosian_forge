from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_datetime(self):
    rng = period_range(start='2012-01-01', periods=10, freq='W-MON')
    ts = Series(range(len(rng)), index=rng)
    dt1 = datetime(2011, 10, 2)
    dt4 = datetime(2012, 4, 20)
    rs = ts[dt1:dt4]
    tm.assert_series_equal(rs, ts)