from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.fixture
def period_index(freqstr):
    """
    A fixture to provide PeriodIndex objects with different frequencies.

    Most PeriodArray behavior is already tested in PeriodIndex tests,
    so here we just test that the PeriodArray behavior matches
    the PeriodIndex behavior.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Period with BDay freq', category=FutureWarning)
        freqstr = freq_to_period_freqstr(1, freqstr)
        pi = pd.period_range(start=Timestamp('2000-01-01'), periods=100, freq=freqstr)
    return pi