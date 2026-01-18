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
def test_date_formatter_callable():

    class _Locator:

        def _get_unit(self):
            return -11

    def callable_formatting_function(dates, _):
        return [dt.strftime('%d-%m//%Y') for dt in dates]
    formatter = mdates.AutoDateFormatter(_Locator())
    formatter.scaled[-10] = callable_formatting_function
    assert formatter([datetime.datetime(2014, 12, 25)]) == ['25-12//2014']