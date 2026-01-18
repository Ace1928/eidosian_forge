import datetime
import dateutil.tz
import dateutil.rrule
import functools
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import rc_context, style
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.ticker as mticker
def test_date2num_dst_pandas(pd):

    def tz_convert(*args):
        return pd.DatetimeIndex.tz_convert(*args).astype(object)
    _test_date2num_dst(pd.date_range, tz_convert)