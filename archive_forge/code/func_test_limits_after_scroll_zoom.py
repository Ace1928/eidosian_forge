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
def test_limits_after_scroll_zoom():
    fig, ax = plt.subplots()
    xlim = (-0.5, 0.5)
    ylim = (-1, 2)
    ax.set_xlim(xlim)
    ax.set_ylim(ymin=ylim[0], ymax=ylim[1])
    ax._set_view_from_bbox((200, 200, 1.0))
    np.testing.assert_allclose(xlim, ax.get_xlim(), atol=1e-16)
    np.testing.assert_allclose(ylim, ax.get_ylim(), atol=1e-16)
    ax._set_view_from_bbox((200, 200, 2.0))
    new_xlim = (-0.3790322580645161, 0.12096774193548387)
    new_ylim = (-0.40625, 1.09375)
    res_xlim = ax.get_xlim()
    res_ylim = ax.get_ylim()
    np.testing.assert_allclose(res_xlim[1] - res_xlim[0], 0.5)
    np.testing.assert_allclose(res_ylim[1] - res_ylim[0], 1.5)
    np.testing.assert_allclose(new_xlim, res_xlim, atol=1e-16)
    np.testing.assert_allclose(new_ylim, res_ylim)
    ax._set_view_from_bbox((200, 200, 0.5))
    res_xlim = ax.get_xlim()
    res_ylim = ax.get_ylim()
    np.testing.assert_allclose(res_xlim[1] - res_xlim[0], 1)
    np.testing.assert_allclose(res_ylim[1] - res_ylim[0], 3)
    np.testing.assert_allclose(xlim, res_xlim, atol=1e-16)
    np.testing.assert_allclose(ylim, res_ylim, atol=1e-16)