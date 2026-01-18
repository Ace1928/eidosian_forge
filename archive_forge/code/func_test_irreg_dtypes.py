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
def test_irreg_dtypes(self):
    idx = [date(2000, 1, 1), date(2000, 1, 5), date(2000, 1, 20)]
    df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), Index(idx, dtype=object))
    _check_plot_works(df.plot)