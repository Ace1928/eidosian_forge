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
def test_box_aspect():
    fig1, ax1 = plt.subplots()
    axtwin = ax1.twinx()
    axtwin.plot([12, 344])
    ax1.set_box_aspect(1)
    assert ax1.get_box_aspect() == 1.0
    fig2, ax2 = plt.subplots()
    ax2.margins(0)
    ax2.plot([0, 2], [6, 8])
    ax2.set_aspect('equal', adjustable='box')
    fig1.canvas.draw()
    fig2.canvas.draw()
    bb1 = ax1.get_position()
    bbt = axtwin.get_position()
    bb2 = ax2.get_position()
    assert_array_equal(bb1.extents, bb2.extents)
    assert_array_equal(bbt.extents, bb2.extents)