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
@pytest.mark.parametrize('delta, expected', [(datetime.timedelta(weeks=52 * 200), ['$\\mathdefault{%d}$' % year for year in range(1990, 2171, 20)]), (datetime.timedelta(days=30), ['$\\mathdefault{1990{-}01{-}%02d}$' % day for day in range(1, 32, 3)]), (datetime.timedelta(hours=20), ['$\\mathdefault{01{-}01\\;%02d}$' % hour for hour in range(0, 21, 2)]), (datetime.timedelta(minutes=10), ['$\\mathdefault{01\\;00{:}%02d}$' % minu for minu in range(0, 11)])])
def test_date_formatter_usetex(delta, expected):
    style.use('default')
    d1 = datetime.datetime(1990, 1, 1)
    d2 = d1 + delta
    locator = mdates.AutoDateLocator(interval_multiples=False)
    locator.create_dummy_axis()
    locator.axis.set_view_interval(mdates.date2num(d1), mdates.date2num(d2))
    formatter = mdates.AutoDateFormatter(locator, usetex=True)
    assert [formatter(loc) for loc in locator()] == expected