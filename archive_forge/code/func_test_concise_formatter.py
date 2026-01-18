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
def test_concise_formatter():

    def _create_auto_date_locator(date1, date2):
        fig, ax = plt.subplots()
        locator = mdates.AutoDateLocator(interval_multiples=True)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylim(date1, date2)
        fig.canvas.draw()
        sts = [st.get_text() for st in ax.get_yticklabels()]
        return sts
    d1 = datetime.datetime(1997, 1, 1)
    results = ([datetime.timedelta(weeks=52 * 200), [str(t) for t in range(1980, 2201, 20)]], [datetime.timedelta(weeks=52), ['1997', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']], [datetime.timedelta(days=141), ['Jan', '15', 'Feb', '15', 'Mar', '15', 'Apr', '15', 'May', '15']], [datetime.timedelta(days=40), ['Jan', '05', '09', '13', '17', '21', '25', '29', 'Feb', '05', '09']], [datetime.timedelta(hours=40), ['Jan-01', '04:00', '08:00', '12:00', '16:00', '20:00', 'Jan-02', '04:00', '08:00', '12:00', '16:00']], [datetime.timedelta(minutes=20), ['00:00', '00:05', '00:10', '00:15', '00:20']], [datetime.timedelta(seconds=40), ['00:00', '05', '10', '15', '20', '25', '30', '35', '40']], [datetime.timedelta(seconds=2), ['59.5', '00:00', '00.5', '01.0', '01.5', '02.0', '02.5']])
    for t_delta, expected in results:
        d2 = d1 + t_delta
        strings = _create_auto_date_locator(d1, d2)
        assert strings == expected