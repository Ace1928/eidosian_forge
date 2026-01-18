from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_dt64ser_sub_datetime_dtype(self, unit):
    ts = Timestamp(datetime(1993, 1, 7, 13, 30, 0))
    dt = datetime(1993, 6, 22, 13, 30)
    ser = Series([ts], dtype=f'M8[{unit}]')
    result = ser - dt
    exp_unit = tm.get_finest_unit(unit, 'us')
    assert result.dtype == f'timedelta64[{exp_unit}]'