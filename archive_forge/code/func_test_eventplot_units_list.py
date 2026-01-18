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
@check_figures_equal(extensions=['png'])
def test_eventplot_units_list(fig_test, fig_ref):
    ts_1 = [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 1, 2), datetime.datetime(2021, 1, 3)]
    ts_2 = [datetime.datetime(2021, 1, 15), datetime.datetime(2021, 1, 16)]
    ax = fig_ref.subplots()
    ax.eventplot(ts_1, lineoffsets=0)
    ax.eventplot(ts_2, lineoffsets=1)
    ax = fig_test.subplots()
    ax.eventplot([ts_1, ts_2])