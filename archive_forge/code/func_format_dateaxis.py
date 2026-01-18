from __future__ import annotations
import functools
from typing import (
import warnings
import numpy as np
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.converter import (
from pandas.tseries.frequencies import (
def format_dateaxis(subplot, freq: BaseOffset, index: DatetimeIndex | PeriodIndex) -> None:
    """
    Pretty-formats the date axis (x-axis).

    Major and minor ticks are automatically set for the frequency of the
    current underlying series.  As the dynamic mode is activated by
    default, changing the limits of the x axis will intelligently change
    the positions of the ticks.
    """
    from matplotlib import pylab
    if isinstance(index, ABCPeriodIndex):
        majlocator = TimeSeries_DateLocator(freq, dynamic_mode=True, minor_locator=False, plot_obj=subplot)
        minlocator = TimeSeries_DateLocator(freq, dynamic_mode=True, minor_locator=True, plot_obj=subplot)
        subplot.xaxis.set_major_locator(majlocator)
        subplot.xaxis.set_minor_locator(minlocator)
        majformatter = TimeSeries_DateFormatter(freq, dynamic_mode=True, minor_locator=False, plot_obj=subplot)
        minformatter = TimeSeries_DateFormatter(freq, dynamic_mode=True, minor_locator=True, plot_obj=subplot)
        subplot.xaxis.set_major_formatter(majformatter)
        subplot.xaxis.set_minor_formatter(minformatter)
        subplot.format_coord = functools.partial(_format_coord, freq)
    elif isinstance(index, ABCTimedeltaIndex):
        subplot.xaxis.set_major_formatter(TimeSeries_TimedeltaFormatter())
    else:
        raise TypeError('index type not supported')
    pylab.draw_if_interactive()