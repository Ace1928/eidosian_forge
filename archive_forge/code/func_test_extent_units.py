import contextlib
from collections import namedtuple
import datetime
from decimal import Decimal
from functools import partial
import inspect
import io
from itertools import product
import platform
from types import SimpleNamespace
import dateutil.tz
import numpy as np
from numpy import ma
from cycler import cycler
import pytest
import matplotlib
import matplotlib as mpl
from matplotlib import rc_context, patheffects
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.font_manager as mfont_manager
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.projections.geo import HammerAxes
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import mpl_toolkits.axisartist as AA  # type: ignore
from numpy.testing import (
from matplotlib.testing.decorators import (
@image_comparison(['extent_units.png'], style='mpl20')
def test_extent_units():
    _, axs = plt.subplots(2, 2)
    date_first = np.datetime64('2020-01-01', 'D')
    date_last = np.datetime64('2020-01-11', 'D')
    arr = [[i + j for i in range(10)] for j in range(10)]
    axs[0, 0].set_title('Date extents on y axis')
    im = axs[0, 0].imshow(arr, origin='lower', extent=[1, 11, date_first, date_last], cmap=mpl.colormaps['plasma'])
    axs[0, 1].set_title('Date extents on x axis (Day of Jan 2020)')
    im = axs[0, 1].imshow(arr, origin='lower', extent=[date_first, date_last, 1, 11], cmap=mpl.colormaps['plasma'])
    axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    im = axs[1, 0].imshow(arr, origin='lower', extent=[date_first, date_last, date_first, date_last], cmap=mpl.colormaps['plasma'])
    axs[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    axs[1, 0].set(xlabel='Day of Jan 2020')
    im = axs[1, 1].imshow(arr, origin='lower', cmap=mpl.colormaps['plasma'])
    im.set_extent([date_last, date_first, date_last, date_first])
    axs[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    axs[1, 1].set(xlabel='Day of Jan 2020')
    with pytest.raises(TypeError, match='set_extent\\(\\) got an unexpected'):
        im.set_extent([2, 12, date_first, date_last], clip=False)