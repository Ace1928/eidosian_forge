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
def test_hist_stacked_stepfilled_geometry():
    bins = [0, 1, 2, 3]
    data_1 = [0, 0, 1, 1, 1, 2]
    data_2 = [0, 1, 2]
    _, _, patches = plt.hist([data_1, data_2], bins=bins, stacked=True, histtype='stepfilled')
    assert len(patches) == 2
    polygon, = patches[0]
    xy = [[0, 0], [0, 2], [1, 2], [1, 3], [2, 3], [2, 1], [3, 1], [3, 0], [2, 0], [2, 0], [1, 0], [1, 0], [0, 0]]
    assert_array_equal(polygon.get_xy(), xy)
    polygon, = patches[1]
    xy = [[0, 2], [0, 3], [1, 3], [1, 4], [2, 4], [2, 2], [3, 2], [3, 1], [2, 1], [2, 3], [1, 3], [1, 2], [0, 2]]
    assert_array_equal(polygon.get_xy(), xy)