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
def test_YearLocator():

    def _create_year_locator(date1, date2, **kwargs):
        locator = mdates.YearLocator(**kwargs)
        locator.create_dummy_axis()
        locator.axis.set_view_interval(mdates.date2num(date1), mdates.date2num(date2))
        return locator
    d1 = datetime.datetime(1990, 1, 1)
    results = ([datetime.timedelta(weeks=52 * 200), {'base': 20, 'month': 1, 'day': 1}, ['1980-01-01 00:00:00+00:00', '2000-01-01 00:00:00+00:00', '2020-01-01 00:00:00+00:00', '2040-01-01 00:00:00+00:00', '2060-01-01 00:00:00+00:00', '2080-01-01 00:00:00+00:00', '2100-01-01 00:00:00+00:00', '2120-01-01 00:00:00+00:00', '2140-01-01 00:00:00+00:00', '2160-01-01 00:00:00+00:00', '2180-01-01 00:00:00+00:00', '2200-01-01 00:00:00+00:00']], [datetime.timedelta(weeks=52 * 200), {'base': 20, 'month': 5, 'day': 16}, ['1980-05-16 00:00:00+00:00', '2000-05-16 00:00:00+00:00', '2020-05-16 00:00:00+00:00', '2040-05-16 00:00:00+00:00', '2060-05-16 00:00:00+00:00', '2080-05-16 00:00:00+00:00', '2100-05-16 00:00:00+00:00', '2120-05-16 00:00:00+00:00', '2140-05-16 00:00:00+00:00', '2160-05-16 00:00:00+00:00', '2180-05-16 00:00:00+00:00', '2200-05-16 00:00:00+00:00']], [datetime.timedelta(weeks=52 * 5), {'base': 20, 'month': 9, 'day': 25}, ['1980-09-25 00:00:00+00:00', '2000-09-25 00:00:00+00:00']])
    for delta, arguments, expected in results:
        d2 = d1 + delta
        locator = _create_year_locator(d1, d2, **arguments)
        assert list(map(str, mdates.num2date(locator()))) == expected