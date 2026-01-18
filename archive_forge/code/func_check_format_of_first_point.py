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
def check_format_of_first_point(ax, expected_string):
    first_line = ax.get_lines()[0]
    first_x = first_line.get_xdata()[0].ordinal
    first_y = first_line.get_ydata()[0]
    assert expected_string == ax.format_coord(first_x, first_y)