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
def test_box_aspect_custom_position():
    fig1, ax1 = plt.subplots()
    ax1.set_position([0.1, 0.1, 0.9, 0.2])
    fig1.canvas.draw()
    ax1.set_box_aspect(1.0)
    fig2, ax2 = plt.subplots()
    ax2.set_box_aspect(1.0)
    fig2.canvas.draw()
    ax2.set_position([0.1, 0.1, 0.9, 0.2])
    fig1.canvas.draw()
    fig2.canvas.draw()
    bb1 = ax1.get_position()
    bb2 = ax2.get_position()
    assert_array_equal(bb1.extents, bb2.extents)