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
def test_concise_formatter_call():
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    assert formatter(19002.0) == '2022'
    assert formatter.format_data_short(19002.0) == '2022-01-10 00:00:00'