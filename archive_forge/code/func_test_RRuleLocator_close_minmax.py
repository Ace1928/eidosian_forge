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
def test_RRuleLocator_close_minmax():
    rrule = mdates.rrulewrapper(dateutil.rrule.SECONDLY, interval=5)
    loc = mdates.RRuleLocator(rrule)
    d1 = datetime.datetime(year=2020, month=1, day=1)
    d2 = datetime.datetime(year=2020, month=1, day=1, microsecond=1)
    expected = ['2020-01-01 00:00:00+00:00', '2020-01-01 00:00:00.000001+00:00']
    assert list(map(str, mdates.num2date(loc.tick_values(d1, d2)))) == expected