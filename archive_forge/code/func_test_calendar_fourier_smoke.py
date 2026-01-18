from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
@pytest.mark.smoke
def test_calendar_fourier_smoke(time_index, forecast_index):
    cf = CalendarFourier(YEAR_END, 2)
    cf.in_sample(time_index)
    steps = 83 if forecast_index is None else len(forecast_index)
    cf.out_of_sample(steps, time_index, forecast_index)
    assert isinstance(cf.order, int)
    assert isinstance(cf.freq, str)
    str(cf)
    repr(cf)
    hash(cf)