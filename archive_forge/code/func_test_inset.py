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
def test_inset():
    """
    Ensure that inset_ax argument is indeed optional
    """
    dx, dy = (0.05, 0.05)
    y, x = np.mgrid[slice(1, 5 + dy, dy), slice(1, 5 + dx, dx)]
    z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, z[:-1, :-1])
    ax.set_aspect(1.0)
    ax.apply_aspect()
    xlim = [1.5, 2.15]
    ylim = [2, 2.5]
    rect = [xlim[0], ylim[0], xlim[1] - xlim[0], ylim[1] - ylim[0]]
    rec, connectors = ax.indicate_inset(bounds=rect)
    assert connectors is None
    fig.canvas.draw()
    xx = np.array([[1.5, 2.0], [2.15, 2.5]])
    assert np.all(rec.get_bbox().get_points() == xx)