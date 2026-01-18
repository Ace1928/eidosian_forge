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
@_new_epoch_decorator
def test_auto_date_locator():

    def _create_auto_date_locator(date1, date2):
        locator = mdates.AutoDateLocator(interval_multiples=False)
        locator.create_dummy_axis()
        locator.axis.set_view_interval(*mdates.date2num([date1, date2]))
        return locator
    d1 = datetime.datetime(1990, 1, 1)
    results = ([datetime.timedelta(weeks=52 * 200), ['1990-01-01 00:00:00+00:00', '2010-01-01 00:00:00+00:00', '2030-01-01 00:00:00+00:00', '2050-01-01 00:00:00+00:00', '2070-01-01 00:00:00+00:00', '2090-01-01 00:00:00+00:00', '2110-01-01 00:00:00+00:00', '2130-01-01 00:00:00+00:00', '2150-01-01 00:00:00+00:00', '2170-01-01 00:00:00+00:00']], [datetime.timedelta(weeks=52), ['1990-01-01 00:00:00+00:00', '1990-02-01 00:00:00+00:00', '1990-03-01 00:00:00+00:00', '1990-04-01 00:00:00+00:00', '1990-05-01 00:00:00+00:00', '1990-06-01 00:00:00+00:00', '1990-07-01 00:00:00+00:00', '1990-08-01 00:00:00+00:00', '1990-09-01 00:00:00+00:00', '1990-10-01 00:00:00+00:00', '1990-11-01 00:00:00+00:00', '1990-12-01 00:00:00+00:00']], [datetime.timedelta(days=141), ['1990-01-05 00:00:00+00:00', '1990-01-26 00:00:00+00:00', '1990-02-16 00:00:00+00:00', '1990-03-09 00:00:00+00:00', '1990-03-30 00:00:00+00:00', '1990-04-20 00:00:00+00:00', '1990-05-11 00:00:00+00:00']], [datetime.timedelta(days=40), ['1990-01-03 00:00:00+00:00', '1990-01-10 00:00:00+00:00', '1990-01-17 00:00:00+00:00', '1990-01-24 00:00:00+00:00', '1990-01-31 00:00:00+00:00', '1990-02-07 00:00:00+00:00']], [datetime.timedelta(hours=40), ['1990-01-01 00:00:00+00:00', '1990-01-01 04:00:00+00:00', '1990-01-01 08:00:00+00:00', '1990-01-01 12:00:00+00:00', '1990-01-01 16:00:00+00:00', '1990-01-01 20:00:00+00:00', '1990-01-02 00:00:00+00:00', '1990-01-02 04:00:00+00:00', '1990-01-02 08:00:00+00:00', '1990-01-02 12:00:00+00:00', '1990-01-02 16:00:00+00:00']], [datetime.timedelta(minutes=20), ['1990-01-01 00:00:00+00:00', '1990-01-01 00:05:00+00:00', '1990-01-01 00:10:00+00:00', '1990-01-01 00:15:00+00:00', '1990-01-01 00:20:00+00:00']], [datetime.timedelta(seconds=40), ['1990-01-01 00:00:00+00:00', '1990-01-01 00:00:05+00:00', '1990-01-01 00:00:10+00:00', '1990-01-01 00:00:15+00:00', '1990-01-01 00:00:20+00:00', '1990-01-01 00:00:25+00:00', '1990-01-01 00:00:30+00:00', '1990-01-01 00:00:35+00:00', '1990-01-01 00:00:40+00:00']], [datetime.timedelta(microseconds=1500), ['1989-12-31 23:59:59.999500+00:00', '1990-01-01 00:00:00+00:00', '1990-01-01 00:00:00.000500+00:00', '1990-01-01 00:00:00.001000+00:00', '1990-01-01 00:00:00.001500+00:00', '1990-01-01 00:00:00.002000+00:00']])
    for t_delta, expected in results:
        d2 = d1 + t_delta
        locator = _create_auto_date_locator(d1, d2)
        assert list(map(str, mdates.num2date(locator()))) == expected
    locator = mdates.AutoDateLocator(interval_multiples=False)
    assert locator.maxticks == {0: 11, 1: 12, 3: 11, 4: 12, 5: 11, 6: 11, 7: 8}
    locator = mdates.AutoDateLocator(maxticks={dateutil.rrule.MONTHLY: 5})
    assert locator.maxticks == {0: 11, 1: 5, 3: 11, 4: 12, 5: 11, 6: 11, 7: 8}
    locator = mdates.AutoDateLocator(maxticks=5)
    assert locator.maxticks == {0: 5, 1: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5}