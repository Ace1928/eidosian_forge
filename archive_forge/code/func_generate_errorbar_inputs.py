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
def generate_errorbar_inputs():
    base_xy = cycler('x', [np.arange(5)]) + cycler('y', [np.ones(5)])
    err_cycler = cycler('err', [1, [1, 1, 1, 1, 1], [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], np.ones(5), np.ones((2, 5)), None])
    xerr_cy = cycler('xerr', err_cycler)
    yerr_cy = cycler('yerr', err_cycler)
    empty = (cycler('x', [[]]) + cycler('y', [[]])) * cycler('xerr', [[], None]) * cycler('yerr', [[], None])
    xerr_only = base_xy * xerr_cy
    yerr_only = base_xy * yerr_cy
    both_err = base_xy * yerr_cy * xerr_cy
    return [*xerr_only, *yerr_only, *both_err, *empty]