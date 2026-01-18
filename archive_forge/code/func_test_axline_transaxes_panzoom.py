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
def test_axline_transaxes_panzoom(fig_test, fig_ref):
    ax = fig_test.subplots()
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    ax.axline((0, 0), slope=1, transform=ax.transAxes)
    ax.axline((0.5, 0.5), slope=2, color='C1', transform=ax.transAxes)
    ax.axline((0.5, 0.5), slope=0, color='C2', transform=ax.transAxes)
    ax.set(xlim=(0, 5), ylim=(0, 10))
    fig_test.set_size_inches(3, 3)
    ax = fig_ref.subplots()
    ax.set(xlim=(0, 5), ylim=(0, 10))
    fig_ref.set_size_inches(3, 3)
    ax.plot([0, 5], [0, 5])
    ax.plot([0, 5], [0, 10], color='C1')
    ax.plot([0, 5], [5, 5], color='C2')