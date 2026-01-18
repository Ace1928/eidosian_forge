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
def test_date_empty():
    fig, ax = plt.subplots()
    ax.xaxis_date()
    fig.draw_without_rendering()
    np.testing.assert_allclose(ax.get_xlim(), [mdates.date2num(np.datetime64('1970-01-01')), mdates.date2num(np.datetime64('1970-01-02'))])
    mdates._reset_epoch_test_example()
    mdates.set_epoch('0000-12-31')
    fig, ax = plt.subplots()
    ax.xaxis_date()
    fig.draw_without_rendering()
    np.testing.assert_allclose(ax.get_xlim(), [mdates.date2num(np.datetime64('1970-01-01')), mdates.date2num(np.datetime64('1970-01-02'))])
    mdates._reset_epoch_test_example()