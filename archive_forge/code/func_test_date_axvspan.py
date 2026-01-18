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
@image_comparison(['date_axvspan.png'])
def test_date_axvspan():
    t0 = datetime.datetime(2000, 1, 20)
    tf = datetime.datetime(2010, 1, 21)
    fig, ax = plt.subplots()
    ax.axvspan(t0, tf, facecolor='blue', alpha=0.25)
    ax.set_xlim(t0 - datetime.timedelta(days=720), tf + datetime.timedelta(days=720))
    fig.autofmt_xdate()