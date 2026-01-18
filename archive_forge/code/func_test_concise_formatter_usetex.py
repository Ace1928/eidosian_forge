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
@pytest.mark.parametrize('t_delta, expected', [(datetime.timedelta(weeks=52 * 200), ['$\\mathdefault{%d}$' % (t,) for t in range(1980, 2201, 20)]), (datetime.timedelta(days=40), ['Jan', '$\\mathdefault{05}$', '$\\mathdefault{09}$', '$\\mathdefault{13}$', '$\\mathdefault{17}$', '$\\mathdefault{21}$', '$\\mathdefault{25}$', '$\\mathdefault{29}$', 'Feb', '$\\mathdefault{05}$', '$\\mathdefault{09}$']), (datetime.timedelta(hours=40), ['Jan$\\mathdefault{{-}01}$', '$\\mathdefault{04{:}00}$', '$\\mathdefault{08{:}00}$', '$\\mathdefault{12{:}00}$', '$\\mathdefault{16{:}00}$', '$\\mathdefault{20{:}00}$', 'Jan$\\mathdefault{{-}02}$', '$\\mathdefault{04{:}00}$', '$\\mathdefault{08{:}00}$', '$\\mathdefault{12{:}00}$', '$\\mathdefault{16{:}00}$']), (datetime.timedelta(seconds=2), ['$\\mathdefault{59.5}$', '$\\mathdefault{00{:}00}$', '$\\mathdefault{00.5}$', '$\\mathdefault{01.0}$', '$\\mathdefault{01.5}$', '$\\mathdefault{02.0}$', '$\\mathdefault{02.5}$'])])
def test_concise_formatter_usetex(t_delta, expected):
    d1 = datetime.datetime(1997, 1, 1)
    d2 = d1 + t_delta
    locator = mdates.AutoDateLocator(interval_multiples=True)
    locator.create_dummy_axis()
    locator.axis.set_view_interval(mdates.date2num(d1), mdates.date2num(d2))
    formatter = mdates.ConciseDateFormatter(locator, usetex=True)
    assert formatter.format_ticks(locator()) == expected