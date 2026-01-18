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
def date_range(start, freq, periods):
    dtstart = dt_tzaware.mk_tzaware(start)
    return [dtstart + i * freq for i in range(periods)]