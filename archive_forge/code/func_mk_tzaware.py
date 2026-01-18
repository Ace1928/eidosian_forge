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
@classmethod
def mk_tzaware(cls, datetime_obj):
    kwargs = {}
    attrs = ('year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'tzinfo')
    for attr in attrs:
        val = getattr(datetime_obj, attr, None)
        if val is not None:
            kwargs[attr] = val
    return cls(**kwargs)