from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.period import (
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.tests.plotting.common import _check_ticks_props
from pandas.tseries.offsets import WeekOfMonth
def test_get_datevalue(self):
    from pandas.plotting._matplotlib.converter import get_datevalue
    assert get_datevalue(None, 'D') is None
    assert get_datevalue(1987, 'Y') == 1987
    assert get_datevalue(Period(1987, 'Y'), 'M') == Period('1987-12', 'M').ordinal
    assert get_datevalue('1/1/1987', 'D') == Period('1987-1-1', 'D').ordinal