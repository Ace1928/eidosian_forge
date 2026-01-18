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
@image_comparison(['RRuleLocator_bounds.png'])
def test_RRuleLocator():
    import matplotlib.testing.jpl_units as units
    units.register()
    t0 = datetime.datetime(1000, 1, 1)
    tf = datetime.datetime(6000, 1, 1)
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_autoscale_on(True)
    ax.plot([t0, tf], [0.0, 1.0], marker='o')
    rrule = mdates.rrulewrapper(dateutil.rrule.YEARLY, interval=500)
    locator = mdates.RRuleLocator(rrule)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    ax.autoscale_view()
    fig.autofmt_xdate()