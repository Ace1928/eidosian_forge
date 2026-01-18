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
def test_concise_converter_stays():
    x = [datetime.datetime(2000, 1, 1), datetime.datetime(2020, 2, 20)]
    y = [0, 1]
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.xaxis.converter = conv = mdates.ConciseDateConverter()
    assert ax.xaxis.units is None
    ax.set_xlim(*x)
    assert ax.xaxis.converter == conv