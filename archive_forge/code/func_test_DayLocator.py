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
def test_DayLocator():
    with pytest.raises(ValueError):
        mdates.DayLocator(interval=-1)
    with pytest.raises(ValueError):
        mdates.DayLocator(interval=-1.5)
    with pytest.raises(ValueError):
        mdates.DayLocator(interval=0)
    with pytest.raises(ValueError):
        mdates.DayLocator(interval=1.3)
    mdates.DayLocator(interval=1.0)