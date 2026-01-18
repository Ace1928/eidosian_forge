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
@check_figures_equal()
def test_errorbar_cycle_ecolor(fig_test, fig_ref):
    x = np.arange(0.1, 4, 0.5)
    y = [np.exp(-x + n) for n in range(4)]
    axt = fig_test.subplots()
    axr = fig_ref.subplots()
    for yi, color in zip(y, ['C0', 'C1', 'C2', 'C3']):
        axt.errorbar(x, yi, yerr=yi * 0.25, linestyle='-', marker='o', ecolor='black')
        axr.errorbar(x, yi, yerr=yi * 0.25, linestyle='-', marker='o', color=color, ecolor='black')