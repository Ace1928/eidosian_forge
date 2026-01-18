import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.fixture(params=[DatetimeIndex, TimedeltaIndex, PeriodIndex])
def non_monotonic_idx(self, request):
    if request.param is DatetimeIndex:
        return DatetimeIndex(['2000-01-04', '2000-01-01', '2000-01-02'])
    elif request.param is PeriodIndex:
        dti = DatetimeIndex(['2000-01-04', '2000-01-01', '2000-01-02'])
        return dti.to_period('D')
    else:
        return TimedeltaIndex(['1 day 00:00:05', '1 day 00:00:01', '1 day 00:00:02'])