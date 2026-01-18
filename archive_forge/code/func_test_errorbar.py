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
@image_comparison(['errorbar_basic', 'errorbar_mixed', 'errorbar_basic'])
def test_errorbar():
    x = np.arange(0.1, 4, 0.5, dtype=np.longdouble)
    y = np.exp(-x)
    yerr = 0.1 + 0.2 * np.sqrt(x)
    xerr = 0.1 + yerr
    fig = plt.figure()
    ax = fig.gca()
    ax.errorbar(x, y, xerr=0.2, yerr=0.4)
    ax.set_title('Simplest errorbars, 0.2 in x, 0.4 in y')
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax = axs[0, 0]
    ax.errorbar(x, y, yerr=yerr, fmt='o')
    ax.set_title('Vert. symmetric')
    ax.locator_params(nbins=4)
    ax = axs[0, 1]
    ax.errorbar(x, y, xerr=xerr, fmt='o', alpha=0.4)
    ax.set_title('Hor. symmetric w/ alpha')
    ax = axs[1, 0]
    ax.errorbar(x, y, yerr=[yerr, 2 * yerr], xerr=[xerr, 2 * xerr], fmt='--o')
    ax.set_title('H, V asymmetric')
    ax = axs[1, 1]
    ax.set_yscale('log')
    ylower = np.maximum(0.01, y - yerr)
    yerr_lower = y - ylower
    ax.errorbar(x, y, yerr=[yerr_lower, 2 * yerr], xerr=xerr, fmt='o', ecolor='g', capthick=2)
    ax.set_title('Mixed sym., log y')
    ax.set_ylim(0.01, 10.0)
    fig.suptitle('Variable errorbars')
    data = {'x': x, 'y': y}
    fig = plt.figure()
    ax = fig.gca()
    ax.errorbar('x', 'y', xerr=0.2, yerr=0.4, data=data)
    ax.set_title('Simplest errorbars, 0.2 in x, 0.4 in y')