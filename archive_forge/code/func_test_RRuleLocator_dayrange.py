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
def test_RRuleLocator_dayrange():
    loc = mdates.DayLocator()
    x1 = datetime.datetime(year=1, month=1, day=1, tzinfo=mdates.UTC)
    y1 = datetime.datetime(year=1, month=1, day=16, tzinfo=mdates.UTC)
    loc.tick_values(x1, y1)