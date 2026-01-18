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
@image_comparison(['hist2d', 'hist2d'], remove_text=True, style='mpl20')
def test_hist2d():
    plt.rcParams['pcolormesh.snap'] = False
    np.random.seed(0)
    x = np.random.randn(100) * 2 + 5
    y = np.random.randn(100) - 2
    fig, ax = plt.subplots()
    ax.hist2d(x, y, bins=10, rasterized=True)
    data = {'x': x, 'y': y}
    fig, ax = plt.subplots()
    ax.hist2d('x', 'y', bins=10, data=data, rasterized=True)