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
def test_date_not_empty():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot([50, 70], [1, 2])
    ax.xaxis.axis_date()
    np.testing.assert_allclose(ax.get_xlim(), [50, 70])