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
@image_comparison(['annotate_across_transforms.png'], style='mpl20', remove_text=True)
def test_annotate_across_transforms():
    x = np.linspace(0, 10, 200)
    y = np.exp(-x) * np.sin(x)
    fig, ax = plt.subplots(figsize=(3.39, 3))
    ax.plot(x, y)
    axins = ax.inset_axes([0.4, 0.5, 0.3, 0.3])
    axins.set_aspect(0.2)
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)
    ax.annotate('', xy=(x[150], y[150]), xycoords=ax.transData, xytext=(1, 0), textcoords=axins.transAxes, arrowprops=dict(arrowstyle='->'))