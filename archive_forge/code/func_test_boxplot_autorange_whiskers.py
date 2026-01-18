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
@image_comparison(['boxplot_autorange_false_whiskers.png', 'boxplot_autorange_true_whiskers.png'], style='default')
def test_boxplot_autorange_whiskers():
    np.random.seed(937)
    x = np.ones(140)
    x = np.hstack([0, x, 2])
    fig1, ax1 = plt.subplots()
    ax1.boxplot([x, x], bootstrap=10000, notch=1)
    ax1.set_ylim((-5, 5))
    fig2, ax2 = plt.subplots()
    ax2.boxplot([x, x], bootstrap=10000, notch=1, autorange=True)
    ax2.set_ylim((-5, 5))