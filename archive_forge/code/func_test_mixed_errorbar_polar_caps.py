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
@image_comparison(['mixed_errorbar_polar_caps'], extensions=['png'], remove_text=True)
def test_mixed_errorbar_polar_caps():
    """
    Mix several polar errorbar use cases in a single test figure.

    It is advisable to position individual points off the grid. If there are
    problems with reproducibility of this test, consider removing grid.
    """
    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    th_sym = [1, 2, 3]
    r_sym = [0.9] * 3
    ax.errorbar(th_sym, r_sym, xerr=0.35, yerr=0.2, fmt='o')
    th_long = [np.pi / 2 + 0.1, np.pi + 0.1]
    r_long = [1.8, 2.2]
    ax.errorbar(th_long, r_long, xerr=0.8 * np.pi, yerr=0.15, fmt='o')
    th_asym = [4 * np.pi / 3 + 0.1, 5 * np.pi / 3 + 0.1, 2 * np.pi - 0.1]
    r_asym = [1.1] * 3
    xerr = [[0.3, 0.3, 0.2], [0.2, 0.3, 0.3]]
    yerr = [[0.35, 0.5, 0.5], [0.5, 0.35, 0.5]]
    ax.errorbar(th_asym, r_asym, xerr=xerr, yerr=yerr, fmt='o')
    th_over = [2.1]
    r_over = [3.1]
    ax.errorbar(th_over, r_over, xerr=10, yerr=0.2, fmt='o')