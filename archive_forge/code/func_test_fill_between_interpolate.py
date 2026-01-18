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
@image_comparison(['fill_between_interpolate'], remove_text=True)
def test_fill_between_interpolate():
    x = np.arange(0.0, 2, 0.02)
    y1 = np.sin(2 * np.pi * x)
    y2 = 1.2 * np.sin(4 * np.pi * x)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(x, y1, x, y2, color='black')
    ax1.fill_between(x, y1, y2, where=y2 >= y1, facecolor='white', hatch='/', interpolate=True)
    ax1.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
    y2 = np.ma.masked_greater(y2, 1.0)
    y2[0] = np.ma.masked
    ax2.plot(x, y1, x, y2, color='black')
    ax2.fill_between(x, y1, y2, where=y2 >= y1, facecolor='green', interpolate=True)
    ax2.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)