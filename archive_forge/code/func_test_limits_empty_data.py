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
@mpl.style.context('default')
@pytest.mark.parametrize('plot_fun', ['scatter', 'plot', 'fill_between'])
@check_figures_equal(extensions=['png'])
def test_limits_empty_data(plot_fun, fig_test, fig_ref):
    x = np.arange('2010-01-01', '2011-01-01', dtype='datetime64[D]')
    ax_test = fig_test.subplots()
    ax_ref = fig_ref.subplots()
    getattr(ax_test, plot_fun)([], [])
    for ax in [ax_test, ax_ref]:
        getattr(ax, plot_fun)(x, range(len(x)), color='C0')