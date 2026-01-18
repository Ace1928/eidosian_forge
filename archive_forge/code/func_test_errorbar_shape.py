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
def test_errorbar_shape():
    fig = plt.figure()
    ax = fig.gca()
    x = np.arange(0.1, 4, 0.5)
    y = np.exp(-x)
    yerr1 = 0.1 + 0.2 * np.sqrt(x)
    yerr = np.vstack((yerr1, 2 * yerr1)).T
    xerr = 0.1 + yerr
    with pytest.raises(ValueError):
        ax.errorbar(x, y, yerr=yerr, fmt='o')
    with pytest.raises(ValueError):
        ax.errorbar(x, y, xerr=xerr, fmt='o')
    with pytest.raises(ValueError):
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='o')