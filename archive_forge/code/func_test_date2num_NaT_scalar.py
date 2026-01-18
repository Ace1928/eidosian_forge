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
@pytest.mark.parametrize('units', ['s', 'ms', 'us', 'ns'])
def test_date2num_NaT_scalar(units):
    tmpl = mdates.date2num(np.datetime64('NaT', units))
    assert np.isnan(tmpl)