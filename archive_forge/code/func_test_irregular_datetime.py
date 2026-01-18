from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_irregular_datetime(self):
    from pandas.plotting._matplotlib.converter import DatetimeConverter
    rng = date_range('1/1/2000', '3/1/2000')
    rng = rng[[0, 1, 2, 3, 5, 9, 10, 11, 12]]
    ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
    _, ax = mpl.pyplot.subplots()
    ax = ser.plot(ax=ax)
    xp = DatetimeConverter.convert(datetime(1999, 1, 1), '', ax)
    ax.set_xlim('1/1/1999', '1/1/2001')
    assert xp == ax.get_xlim()[0]
    _check_ticks_props(ax, xrot=30)