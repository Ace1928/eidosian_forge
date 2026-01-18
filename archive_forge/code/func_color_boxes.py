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
def color_boxes(fig, ax):
    """
    Helper for the tests below that test the extents of various axes elements
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbaxis = []
    for nn, axx in enumerate([ax.xaxis, ax.yaxis]):
        bb = axx.get_tightbbox(renderer)
        if bb:
            axisr = mpatches.Rectangle((bb.x0, bb.y0), width=bb.width, height=bb.height, linewidth=0.7, edgecolor='y', facecolor='none', transform=None, zorder=3)
            fig.add_artist(axisr)
        bbaxis += [bb]
    bbspines = []
    for nn, a in enumerate(['bottom', 'top', 'left', 'right']):
        bb = ax.spines[a].get_window_extent(renderer)
        spiner = mpatches.Rectangle((bb.x0, bb.y0), width=bb.width, height=bb.height, linewidth=0.7, edgecolor='green', facecolor='none', transform=None, zorder=3)
        fig.add_artist(spiner)
        bbspines += [bb]
    bb = ax.get_window_extent()
    rect2 = mpatches.Rectangle((bb.x0, bb.y0), width=bb.width, height=bb.height, linewidth=1.5, edgecolor='magenta', facecolor='none', transform=None, zorder=2)
    fig.add_artist(rect2)
    bbax = bb
    bb2 = ax.get_tightbbox(renderer)
    rect2 = mpatches.Rectangle((bb2.x0, bb2.y0), width=bb2.width, height=bb2.height, linewidth=3, edgecolor='red', facecolor='none', transform=None, zorder=1)
    fig.add_artist(rect2)
    bbtb = bb2
    return (bbaxis, bbspines, bbax, bbtb)