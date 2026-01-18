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
@image_comparison(['date_axhspan.png'])
def test_date_axhspan():
    t0 = datetime.datetime(2009, 1, 20)
    tf = datetime.datetime(2009, 1, 21)
    fig, ax = plt.subplots()
    ax.axhspan(t0, tf, facecolor='blue', alpha=0.25)
    ax.set_ylim(t0 - datetime.timedelta(days=5), tf + datetime.timedelta(days=5))
    fig.subplots_adjust(left=0.25)