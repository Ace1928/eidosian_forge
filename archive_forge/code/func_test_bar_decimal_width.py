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
def test_bar_decimal_width(fig_test, fig_ref):
    x = [1.5, 8.4, 5.3, 4.2]
    y = [1.1, 2.2, 3.3, 4.4]
    w0 = [0.7, 1.45, 1, 2]
    w = [Decimal(i) for i in w0]
    ax = fig_test.subplots()
    ax.bar(x, y, width=w, align='center')
    ax = fig_ref.subplots()
    ax.bar(x, y, width=w0, align='center')