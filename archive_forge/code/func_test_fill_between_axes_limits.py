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
def test_fill_between_axes_limits():
    fig, ax = plt.subplots()
    x = np.arange(0, 4 * np.pi, 0.01)
    y = 0.1 * np.sin(x)
    threshold = 0.075
    ax.plot(x, y, color='black')
    original_lims = (ax.get_xlim(), ax.get_ylim())
    ax.axhline(threshold, color='green', lw=2, alpha=0.7)
    ax.fill_between(x, 0, 1, where=y > threshold, color='green', alpha=0.5, transform=ax.get_xaxis_transform())
    assert (ax.get_xlim(), ax.get_ylim()) == original_lims