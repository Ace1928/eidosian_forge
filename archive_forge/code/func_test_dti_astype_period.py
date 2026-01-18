from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dti_astype_period(self):
    idx = DatetimeIndex([NaT, '2011-01-01', '2011-02-01'], name='idx')
    res = idx.astype('period[M]')
    exp = PeriodIndex(['NaT', '2011-01', '2011-02'], freq='M', name='idx')
    tm.assert_index_equal(res, exp)
    res = idx.astype('period[3M]')
    exp = PeriodIndex(['NaT', '2011-01', '2011-02'], freq='3M', name='idx')
    tm.assert_index_equal(res, exp)