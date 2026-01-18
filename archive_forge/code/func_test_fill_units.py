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
@image_comparison(['fill_units.png'], savefig_kwarg={'dpi': 60})
def test_fill_units():
    import matplotlib.testing.jpl_units as units
    units.register()
    t = units.Epoch('ET', dt=datetime.datetime(2009, 4, 27))
    value = 10.0 * units.deg
    day = units.Duration('ET', 24.0 * 60.0 * 60.0)
    dt = np.arange('2009-04-27', '2009-04-29', dtype='datetime64[D]')
    dtn = mdates.date2num(dt)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot([t], [value], yunits='deg', color='red')
    ind = [0, 0, 1, 1]
    ax1.fill(dtn[ind], [0.0, 0.0, 90.0, 0.0], 'b')
    ax2.plot([t], [value], yunits='deg', color='red')
    ax2.fill([t, t, t + day, t + day], [0.0, 0.0, 90.0, 0.0], 'b')
    ax3.plot([t], [value], yunits='deg', color='red')
    ax3.fill(dtn[ind], [0 * units.deg, 0 * units.deg, 90 * units.deg, 0 * units.deg], 'b')
    ax4.plot([t], [value], yunits='deg', color='red')
    ax4.fill([t, t, t + day, t + day], [0 * units.deg, 0 * units.deg, 90 * units.deg, 0 * units.deg], facecolor='blue')
    fig.autofmt_xdate()