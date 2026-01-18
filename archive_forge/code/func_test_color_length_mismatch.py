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
def test_color_length_mismatch():
    N = 5
    x, y = (np.arange(N), np.arange(N))
    colors = np.arange(N + 1)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.scatter(x, y, c=colors)
    with pytest.warns(match='argument looks like a single numeric RGB'):
        ax.scatter(x, y, c=(0.5, 0.5, 0.5))
    ax.scatter(x, y, c=[(0.5, 0.5, 0.5)] * N)