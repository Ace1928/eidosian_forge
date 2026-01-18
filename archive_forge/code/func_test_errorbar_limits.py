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
@image_comparison(['errorbar_limits'])
def test_errorbar_limits():
    x = np.arange(0.5, 5.5, 0.5)
    y = np.exp(-x)
    xerr = 0.1
    yerr = 0.2
    ls = 'dotted'
    fig, ax = plt.subplots()
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, ls=ls, color='blue')
    uplims = np.zeros_like(x)
    uplims[[1, 5, 9]] = True
    ax.errorbar(x, y + 0.5, xerr=xerr, yerr=yerr, uplims=uplims, ls=ls, color='green')
    lolims = np.zeros_like(x)
    lolims[[2, 4, 8]] = True
    ax.errorbar(x, y + 1.0, xerr=xerr, yerr=yerr, lolims=lolims, ls=ls, color='red')
    ax.errorbar(x, y + 1.5, marker='o', ms=8, xerr=xerr, yerr=yerr, lolims=lolims, uplims=uplims, ls=ls, color='magenta')
    xerr = 0.2
    yerr = np.full_like(x, 0.2)
    yerr[[3, 6]] = 0.3
    xlolims = lolims
    xuplims = uplims
    lolims = np.zeros_like(x)
    uplims = np.zeros_like(x)
    lolims[[6]] = True
    uplims[[3]] = True
    ax.errorbar(x, y + 2.1, marker='o', ms=8, xerr=xerr, yerr=yerr, xlolims=xlolims, xuplims=xuplims, uplims=uplims, lolims=lolims, ls='none', mec='blue', capsize=0, color='cyan')
    ax.set_xlim((0, 5.5))
    ax.set_title('Errorbar upper and lower limits')