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
def test_broken_barh_timedelta():
    """Check that timedelta works as x, dx pair for this method."""
    fig, ax = plt.subplots()
    d0 = datetime.datetime(2018, 11, 9, 0, 0, 0)
    pp = ax.broken_barh([(d0, datetime.timedelta(hours=1))], [1, 2])
    assert pp.get_paths()[0].vertices[0, 0] == mdates.date2num(d0)
    assert pp.get_paths()[0].vertices[2, 0] == mdates.date2num(d0) + 1 / 24