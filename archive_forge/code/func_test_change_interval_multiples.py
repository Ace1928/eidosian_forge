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
def test_change_interval_multiples():
    plt.rcParams['date.interval_multiples'] = False
    dates = np.arange('2020-01-10', '2020-05-01', dtype='datetime64[D]')
    fig, ax = plt.subplots()
    ax.plot(dates, np.arange(len(dates)))
    fig.canvas.draw()
    assert ax.get_xticklabels()[0].get_text() == 'Jan 10 2020'
    assert ax.get_xticklabels()[1].get_text() == 'Jan 24 2020'
    plt.rcParams['date.interval_multiples'] = 'True'
    fig, ax = plt.subplots()
    ax.plot(dates, np.arange(len(dates)))
    fig.canvas.draw()
    assert ax.get_xticklabels()[0].get_text() == 'Jan 15 2020'
    assert ax.get_xticklabels()[1].get_text() == 'Feb 01 2020'