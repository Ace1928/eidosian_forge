from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_list_periods(self):
    rng = period_range(start='2012-01-01', periods=10, freq='D')
    ts = Series(range(len(rng)), index=rng)
    exp = ts.iloc[[1]]
    tm.assert_series_equal(ts[[Period('2012-01-02', freq='D')]], exp)